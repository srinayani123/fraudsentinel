"""
PySpark feature engineering pipeline for IEEE-CIS Fraud Detection.

Reads:    data/raw/train_transaction.csv + train_identity.csv
Writes:   data/features/engineered.parquet

Feature engineering covers:
- Join transaction + identity data
- Velocity features per card (txn count / sum amount over rolling windows)
- Behavioral aggregates per card (mean/std amount, distinct merchants)
- Categorical encoding (frequency encoding for high-cardinality)
- Time-based features (hour of day, day of week)
- Email/domain risk features
- Missing value indicators

Output is partitioned Parquet, ready for downstream model training.
"""

import os
import sys

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window

from src.utils.config import FEATURES_DIR, RAW_DIR
from src.utils.logging import get_logger

logger = get_logger(__name__)


def get_spark():
    """Create a Spark session sized for a typical laptop. Tune for prod."""
    return (
        SparkSession.builder.appName("FraudSentinel-FeatureEngineering")
        .config("spark.driver.memory", "8g")
        .config("spark.executor.memory", "8g")
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )


def load_raw(spark):
    """Load and join transaction + identity tables."""
    logger.info("Reading transaction table...")
    txn = spark.read.csv(
        str(RAW_DIR / "train_transaction.csv"),
        header=True,
        inferSchema=True,
    )
    logger.info(f"  Transactions: {txn.count():,} rows, {len(txn.columns)} cols")

    logger.info("Reading identity table...")
    ident = spark.read.csv(
        str(RAW_DIR / "train_identity.csv"),
        header=True,
        inferSchema=True,
    )
    logger.info(f"  Identity:     {ident.count():,} rows, {len(ident.columns)} cols")

    logger.info("Left-joining identity onto transactions...")
    df = txn.join(ident, on="TransactionID", how="left")
    logger.info(f"  Joined:       {df.count():,} rows, {len(df.columns)} cols")
    return df


def add_time_features(df):
    """TransactionDT is seconds since reference. Derive hour/day."""
    logger.info("Adding time features...")
    df = df.withColumn("txn_hour", (F.col("TransactionDT") / 3600).cast("int") % 24)
    df = df.withColumn("txn_day", (F.col("TransactionDT") / 86400).cast("int"))
    df = df.withColumn("txn_dayofweek", F.col("txn_day") % 7)
    df = df.withColumn(
        "is_night",
        ((F.col("txn_hour") >= 22) | (F.col("txn_hour") <= 5)).cast("int"),
    )
    return df


def add_velocity_features(df):
    """Per-card transaction velocity over rolling windows.

    Uses card1 as the card identifier (most common per the dataset description).
    Windows: prior 1 hour, 24 hours, 7 days.
    """
    logger.info("Adding velocity features per card1...")

    df = df.withColumn("ts_seconds", F.col("TransactionDT").cast("long"))

    base_window = Window.partitionBy("card1").orderBy(F.col("ts_seconds"))

    for hours, label in [(1, "1h"), (24, "24h"), (168, "7d")]:
        seconds = hours * 3600
        w = base_window.rangeBetween(-seconds, -1)  # exclude current row
        df = df.withColumn(
            f"card1_txn_count_{label}",
            F.count("TransactionID").over(w),
        )
        df = df.withColumn(
            f"card1_amt_sum_{label}",
            F.coalesce(F.sum("TransactionAmt").over(w), F.lit(0.0)),
        )
        df = df.withColumn(
            f"card1_amt_max_{label}",
            F.coalesce(F.max("TransactionAmt").over(w), F.lit(0.0)),
        )

    # Time since last transaction
    df = df.withColumn(
        "card1_seconds_since_last",
        F.coalesce(
            F.col("ts_seconds") - F.lag("ts_seconds").over(base_window),
            F.lit(-1),
        ),
    )
    return df


def add_behavioral_features(df):
    """Per-card aggregates: mean/std amount, distinct merchants/products."""
    logger.info("Adding per-card behavioral aggregates...")

    agg = df.groupBy("card1").agg(
        F.mean("TransactionAmt").alias("card1_amt_mean"),
        F.stddev("TransactionAmt").alias("card1_amt_std"),
        F.count("TransactionID").alias("card1_total_txns"),
        F.countDistinct("ProductCD").alias("card1_distinct_products"),
        F.countDistinct("addr1").alias("card1_distinct_addr1"),
    )

    df = df.join(agg, on="card1", how="left")

    # Z-score of current amount vs card mean
    df = df.withColumn(
        "card1_amt_zscore",
        F.when(
            F.col("card1_amt_std") > 0,
            (F.col("TransactionAmt") - F.col("card1_amt_mean")) / F.col("card1_amt_std"),
        ).otherwise(0.0),
    )
    return df


def add_email_features(df):
    """Email domain features (P_emaildomain, R_emaildomain)."""
    logger.info("Adding email-domain features...")

    high_risk_domains = ["protonmail.com", "mail.com", "outlook.com"]

    for col in ["P_emaildomain", "R_emaildomain"]:
        df = df.withColumn(
            f"{col}_isnull",
            F.col(col).isNull().cast("int"),
        )
        df = df.withColumn(
            f"{col}_is_highrisk",
            F.when(F.col(col).isin(high_risk_domains), 1).otherwise(0),
        )

    df = df.withColumn(
        "emails_match",
        F.when(
            (F.col("P_emaildomain").isNotNull())
            & (F.col("P_emaildomain") == F.col("R_emaildomain")),
            1,
        ).otherwise(0),
    )
    return df


def select_final_columns(df):
    """Pick the columns we'll use for modeling.

    The IEEE-CIS V/C/D/M columns are anonymized but informative — we keep
    a curated subset to keep memory manageable for the demo.
    """
    keep = [
        # identifiers
        "TransactionID",
        "TransactionDT",
        "isFraud",
        # amount + product
        "TransactionAmt",
        "ProductCD",
        # card features
        "card1",
        "card2",
        "card3",
        "card4",
        "card5",
        "card6",
        # address
        "addr1",
        "addr2",
        "dist1",
        "dist2",
        # email
        "P_emaildomain",
        "R_emaildomain",
        "P_emaildomain_isnull",
        "R_emaildomain_isnull",
        "P_emaildomain_is_highrisk",
        "R_emaildomain_is_highrisk",
        "emails_match",
        # time
        "txn_hour",
        "txn_day",
        "txn_dayofweek",
        "is_night",
        # velocity
        "card1_txn_count_1h",
        "card1_txn_count_24h",
        "card1_txn_count_7d",
        "card1_amt_sum_1h",
        "card1_amt_sum_24h",
        "card1_amt_sum_7d",
        "card1_amt_max_24h",
        "card1_seconds_since_last",
        # behavioral
        "card1_amt_mean",
        "card1_amt_std",
        "card1_amt_zscore",
        "card1_total_txns",
        "card1_distinct_products",
        "card1_distinct_addr1",
        # selected anonymized features
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "C7",
        "C8",
        "C9",
        "C10",
        "C11",
        "C12",
        "C13",
        "C14",
        "D1",
        "D2",
        "D3",
        "D4",
        "D10",
        "D15",
        "V1",
        "V12",
        "V14",
        "V20",
        "V30",
        "V40",
        "V50",
        "V70",
        "V90",
        "V100",
        "V130",
        "V160",
        "V200",
        "V250",
        "V300",
    ]

    available = [c for c in keep if c in df.columns]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        logger.warning(f"Skipping {len(missing)} columns not in dataset: {missing[:5]}...")
    return df.select(*available)


def main():
    if not (RAW_DIR / "train_transaction.csv").exists():
        logger.error(
            f"Raw data not found in {RAW_DIR}. Run download first:\n"
            f"  python -m src.data_ingestion.download_ieee_cis"
        )
        sys.exit(1)

    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")

    try:
        df = load_raw(spark)
        df = add_time_features(df)
        df = add_velocity_features(df)
        df = add_behavioral_features(df)
        df = add_email_features(df)
        df = select_final_columns(df)

        logger.info(f"Final schema: {len(df.columns)} columns")

        # Write Parquet
        out_path = FEATURES_DIR / "engineered.parquet"
        logger.info(f"Writing engineered features to {out_path}...")
        # Single file for easy downstream pandas read
        df.coalesce(1).write.mode("overwrite").parquet(str(out_path))

        # The above writes a directory of part files; rename into place
        # Pandas reads parquet directories fine, so we leave it as-is.
        n_rows = df.count()
        n_fraud = df.filter(F.col("isFraud") == 1).count()
        logger.info(f"✅ Wrote {n_rows:,} rows ({n_fraud:,} fraud, {n_fraud/n_rows:.4%})")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
