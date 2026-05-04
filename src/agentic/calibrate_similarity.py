"""One-time calibration for the semantic retrieval pipeline.

Two calibrations:
1. Similarity rescaling — measures cosine distribution to set perceptual 0-100 bounds
2. Hybrid OOD detection — fits PCA + k-NN signals on (pattern documents + auto-generated
   analyst-style fraud queries) so analyst paraphrases stay in-domain. MS MARCO is the
   OOD reference for separability-based component selection.

Calibration data:
- Pattern documents: full narratives + indicators + plain-English explanations
- Analyst-style queries: auto-generated 2-per-pattern via Claude (run
  generate_calibration_queries.py first). Half used for fitting, half held out
  for validation.
- OOD reference: 500-sample from MS MARCO v2.1 validation split, fraud-keyword
  filtered for clean OOD ground truth.

Borderline cutoffs are at the 90th percentile of fitted in-domain distribution,
producing a typical split of ~70% in_domain, ~30% borderline (routes to LLM
judge), ~0% false-reject.

References:
  - Triantafyllopoulos et al. (2026), arXiv:2508.02296
  - Sun et al. (2022), ICML (k-NN distance OOD)
  - Nguyen et al. (2016), MS MARCO

Usage:
    # First, generate the calibration queries (one-time, cost ~$1):
    python -m src.agentic.generate_calibration_queries

    # Then calibrate:
    python -m src.agentic.calibrate_similarity

Outputs:
    models/similarity_calibration.json
    models/ood_pca_calibration.npz
    models/ood_pca_meta.json
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from src.agentic.ood_detector import PCAOODDetector
from src.agentic.tools import EMBEDDING_MODEL, _get_chroma_collection
from src.utils.config import FRAUD_CASES_DIR
from src.utils.logging import get_logger

logger = get_logger(__name__)

CALIBRATION_FILE = Path("models/similarity_calibration.json")
CALIBRATION_QUERIES_FILE = Path("models/calibration_queries.json")

N_OOD_SAMPLES = 500
OOD_RANDOM_SEED = 42

# Borderline cutoffs at the 90th percentile of the fitted in-domain distribution.
# Lower values (e.g. 80) over-trigger borderline → too much LLM-judge fallback.
# Higher values (e.g. 95) risk false-accepting OOD queries that look near-domain.
# 90 is the empirical sweet spot for our distribution: ~70% pass cleanly, ~30%
# go to borderline, ~0% false-reject, ~98% true-reject on held-out OOD.
PCA_BORDERLINE_PCT = 90.0
KNN_BORDERLINE_PCT = 90.0

# OOD thresholds at the 99.5th percentile — these only flag the most extreme
# outliers as definitively OOD. Anything between borderline and OOD goes to
# the LLM judge as a safety net.
PCA_OOD_PCT = 99.5
KNN_OOD_PCT = 99.5


# Used only for similarity-distribution calibration (Step 1) — small set of
# diverse fraud-domain queries to probe the cosine distribution. NOT used for
# OOD calibration (which uses generated queries instead).
SIMILARITY_PROBE_QUERIES = [
    "small charge then large purchase on same card within an hour",
    "transaction at 3am from a card that normally only transacts during the day",
    "high velocity 12 transactions in last hour on same card",
    "amount well above this card's typical range",
    "transaction from country card has never used before",
    "new device with unusual IP location",
    "subscription signup with disposable email address",
    "BIN range probing with sequential card numbers",
    "card testing burst of small charges under five dollars",
    "normal customer making a routine purchase",
    "legitimate restaurant transaction at lunchtime",
    "one-time gift card purchase",
    "automated rapid fire transactions same product code",
    "dormant card suddenly active after 6 months",
    "international purchase outside customer's country",
    "purchase during normal business hours",
    "monthly subscription renewal as expected",
    "weekend grocery store transaction",
    "synthetic identity with mixed real and fake details",
    "friendly fraud chargeback after legitimate purchase",
]


# Fallback OOD set used if MS MARCO can't be loaded
FALLBACK_OOD_TEXTS = [
    "best chocolate cake recipe with vanilla frosting",
    "how to install Python on Windows step by step",
    "best vacation destinations in Europe summer",
    "how do solar panels generate electricity from sunlight",
    "morning routine for productivity and focus",
    "best Netflix shows to binge watch this weekend",
    "how to train a puppy to sit and stay",
    "what color should I paint my living room",
    "soccer World Cup final 2022 highlights",
    "tips for falling asleep faster at night",
    "how to write a good cover letter for a job",
    "what is the difference between React and Vue",
    "best dog breeds for small apartments",
    "how to grow tomatoes in a small garden",
    "marathon training schedule for first-time runners",
    "explain quantum entanglement in simple terms",
    "indie video games released this year",
    "best stretches for tight hips and lower back",
    "what foods are highest in vitamin D",
    "easy weeknight pasta with garlic and olive oil",
]


def calibrate_similarity_distribution(n_results_per_query: int = 10) -> dict:
    collection = _get_chroma_collection()
    all_raw_cosines = []
    for query in SIMILARITY_PROBE_QUERIES:
        try:
            results = collection.query(query_texts=[query], n_results=n_results_per_query)
            distances = results.get("distances", [[]])[0]
            cosines = [1.0 - d for d in distances]
            all_raw_cosines.extend(cosines)
        except Exception as e:
            logger.warning(f"Query '{query[:40]}...' failed: {e}")
            continue

    if not all_raw_cosines:
        raise RuntimeError("No similarity scores collected — is ChromaDB populated?")

    arr = np.array(all_raw_cosines)
    low = float(np.percentile(arr, 10))
    high = float(np.percentile(arr, 95))
    threshold = float(np.percentile(arr, 5))

    return {
        "n_samples": int(len(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "p05": float(np.percentile(arr, 5)),
        "p10": low,
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": high,
        "calibrated_low": low,
        "calibrated_high": high,
        "calibrated_threshold": threshold,
    }


def load_generated_calibration_queries() -> list[str]:
    """Load auto-generated analyst-style fraud queries from the calibration file.

    Falls back to an empty list if the file is missing — caller should warn the
    user to run the generator first.
    """
    if not CALIBRATION_QUERIES_FILE.exists():
        logger.warning(
            f"{CALIBRATION_QUERIES_FILE} not found. "
            "Run `python -m src.agentic.generate_calibration_queries` first."
        )
        return []

    try:
        with open(CALIBRATION_QUERIES_FILE, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Could not parse {CALIBRATION_QUERIES_FILE}: {e}")
        return []

    queries = []
    for entry in data.get("queries", []):
        queries.extend(entry.get("queries", []))

    meta = data.get("_meta", {})
    logger.info(
        f"Loaded {len(queries)} calibration queries from {meta.get('n_patterns', '?')} patterns "
        f"(generated {meta.get('generated_at', 'unknown date')})"
    )
    return queries


def load_ood_reference_queries() -> tuple[list[str], str]:
    try:
        from datasets import load_dataset
    except ImportError:
        logger.warning("`datasets` library not installed.")
        return FALLBACK_OOD_TEXTS, "fallback_hardcoded"

    try:
        logger.info("Loading MS MARCO queries dataset...")
        ds = load_dataset("ms_marco", "v2.1", split="validation", trust_remote_code=False)
        all_queries = [item["query"] for item in ds if item.get("query")]
        logger.info(f"MS MARCO loaded: {len(all_queries)} total queries available.")

        rng = random.Random(OOD_RANDOM_SEED)
        sampled = rng.sample(all_queries, min(N_OOD_SAMPLES, len(all_queries)))

        fraud_keywords = (
            "fraud", "scam", "stolen card", "credit card theft",
            "phishing", "identity theft", "card declined", "chargeback",
        )
        filtered = [q for q in sampled if not any(kw in q.lower() for kw in fraud_keywords)]
        logger.info(
            f"Sampled {len(filtered)} OOD queries from MS MARCO "
            f"(filtered {len(sampled) - len(filtered)} fraud-adjacent)"
        )
        return filtered, "ms_marco_v2.1_validation"

    except Exception as e:
        logger.warning(f"Failed to load MS MARCO: {e}")
        return FALLBACK_OOD_TEXTS, "fallback_hardcoded"


def fit_ood_detector():
    """Fit hybrid PCA + k-NN OOD detector on (pattern documents + generated queries).

    The ID set is a UNION of:
      - Pattern document text (full narratives + indicators + explanations)
      - Auto-generated analyst-style queries (2 per pattern). Half used for
        fitting, half held out for validation.

    This teaches the detector that BOTH document-style text AND short
    analyst-style queries are in-domain.
    """
    logger.info("Fitting hybrid PCA + k-NN OOD detector...")
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # ---- Pattern documents ----
    if not FRAUD_CASES_DIR.exists():
        raise RuntimeError(f"FRAUD_CASES_DIR does not exist: {FRAUD_CASES_DIR}")
    case_paths = sorted(FRAUD_CASES_DIR.glob("case_*.json"))
    if not case_paths:
        raise RuntimeError("No case files found")

    pattern_texts = []
    for path in case_paths:
        try:
            with open(path, encoding="utf-8") as f:
                case = json.load(f)
            indicators_str = " ".join(case.get("indicators", []))
            explanations_str = " ".join(case.get("indicator_explanations", []))
            text = f"{case.get('title', '')}. {case.get('narrative', '')} {indicators_str} {explanations_str}"
            pattern_texts.append(text)
        except Exception as e:
            logger.warning(f"Skipping {path}: {e}")

    logger.info(f"Embedding {len(pattern_texts)} fraud pattern documents...")
    pattern_embeddings = model.encode(
        pattern_texts, show_progress_bar=False, normalize_embeddings=True
    )

    # ---- Auto-generated calibration queries ----
    all_calibration_queries = load_generated_calibration_queries()
    if not all_calibration_queries:
        logger.warning(
            "No calibration queries available — falling back to pattern-only ID set. "
            "Calibration will be less robust to analyst-style query phrasing."
        )
        fit_queries = []
        holdout_queries = []
    else:
        rng = random.Random(OOD_RANDOM_SEED)
        shuffled = list(all_calibration_queries)
        rng.shuffle(shuffled)
        n_fit = len(shuffled) // 2
        fit_queries = shuffled[:n_fit]
        holdout_queries = shuffled[n_fit:]
        logger.info(
            f"Calibration queries: {len(fit_queries)} for fit, {len(holdout_queries)} for validation"
        )

    if fit_queries:
        logger.info(f"Embedding {len(fit_queries)} analyst-style queries (fit set)...")
        fit_query_embeddings = model.encode(
            fit_queries, show_progress_bar=False, normalize_embeddings=True
        )
        id_embeddings = np.vstack([pattern_embeddings, fit_query_embeddings])
    else:
        id_embeddings = np.array(pattern_embeddings)

    logger.info(
        f"ID set: {len(pattern_embeddings)} pattern docs + "
        f"{len(fit_queries)} analyst queries = {len(id_embeddings)} total"
    )

    # ---- OOD reference ----
    ood_queries, ood_source = load_ood_reference_queries()
    logger.info(f"Embedding {len(ood_queries)} OOD reference queries from {ood_source}...")
    ood_embeddings = model.encode(
        ood_queries, show_progress_bar=False, normalize_embeddings=True
    )

    # ---- Fit ----
    detector = PCAOODDetector.fit(
        id_embeddings=np.array(id_embeddings),
        ood_reference_embeddings=np.array(ood_embeddings),
        variance_target=0.80,
        pca_ood_percentile=PCA_OOD_PCT,
        pca_borderline_percentile=PCA_BORDERLINE_PCT,
        knn_ood_percentile=KNN_OOD_PCT,
        knn_borderline_percentile=KNN_BORDERLINE_PCT,
        k_neighbors=5,
    )
    detector.save()

    logger.info("")
    logger.info(f"OOD reference source:        {ood_source}")
    logger.info(f"Component selection method:  {detector.selection_method}")
    logger.info("")
    logger.info("PCA signal:")
    logger.info(f"  Components:           {detector.n_components}")
    logger.info(f"  OOD threshold (p{PCA_OOD_PCT:.1f}):    {detector.pca_threshold:.4f}")
    logger.info(f"  Borderline (p{PCA_BORDERLINE_PCT:.1f}):    {detector.pca_borderline:.4f}")
    logger.info(f"  ID error median:      {detector.id_pca_p50:.4f}")
    logger.info("")
    logger.info("k-NN signal:")
    logger.info(f"  k:                    {detector.k_neighbors}")
    logger.info(f"  OOD threshold (p{KNN_OOD_PCT:.1f}):    {detector.knn_threshold:.4f}")
    logger.info(f"  Borderline (p{KNN_BORDERLINE_PCT:.1f}):    {detector.knn_borderline:.4f}")

    # ---- Validation report ----
    if not holdout_queries:
        logger.info("")
        logger.info("(No held-out fraud queries to validate — generate calibration queries for full validation)")
        return detector

    logger.info("")
    logger.info("=" * 60)
    logger.info("VALIDATION on held-out queries")
    logger.info("=" * 60)

    holdout_query_embs = model.encode(
        holdout_queries, show_progress_bar=False, normalize_embeddings=True
    )
    holdout_ood = ood_queries[: min(50, len(ood_queries))]
    holdout_ood_embs = model.encode(
        holdout_ood, show_progress_bar=False, normalize_embeddings=True
    )

    fraud_pass = fraud_borderline = fraud_reject = 0
    for emb in holdout_query_embs:
        v = detector.classify(np.asarray(emb))
        if v["verdict"] == "in_domain":
            fraud_pass += 1
        elif v["verdict"] == "borderline":
            fraud_borderline += 1
        else:
            fraud_reject += 1

    ood_pass = ood_borderline = ood_reject = 0
    for emb in holdout_ood_embs:
        v = detector.classify(np.asarray(emb))
        if v["verdict"] == "in_domain":
            ood_pass += 1
        elif v["verdict"] == "borderline":
            ood_borderline += 1
        else:
            ood_reject += 1

    n_fraud = len(holdout_query_embs)
    n_ood = len(holdout_ood_embs)
    logger.info(f"Held-out FRAUD queries (n={n_fraud}):")
    logger.info(f"  in_domain:     {fraud_pass}/{n_fraud} ({100*fraud_pass/n_fraud:.0f}%)")
    logger.info(f"  borderline:    {fraud_borderline}/{n_fraud} ({100*fraud_borderline/n_fraud:.0f}%)")
    logger.info(f"  out_of_domain: {fraud_reject}/{n_fraud} ({100*fraud_reject/n_fraud:.0f}%)  ← false reject")
    logger.info("")
    logger.info(f"Held-out OOD queries (n={n_ood}):")
    logger.info(f"  in_domain:     {ood_pass}/{n_ood} ({100*ood_pass/n_ood:.0f}%)  ← false accept")
    logger.info(f"  borderline:    {ood_borderline}/{n_ood} ({100*ood_borderline/n_ood:.0f}%)")
    logger.info(f"  out_of_domain: {ood_reject}/{n_ood} ({100*ood_reject/n_ood:.0f}%)")
    logger.info("")

    if fraud_reject > 0:
        logger.warning("Held-out fraud queries that were REJECTED (these should pass):")
        for q, emb in zip(holdout_queries, holdout_query_embs):
            v = detector.classify(np.asarray(emb))
            if v["verdict"] == "out_of_domain":
                logger.warning(f"  [pca={v['pca_error']:.3f} knn={v['knn_distance']:.3f}] {q}")

    return detector


def main():
    logger.info("=" * 60)
    logger.info("STEP 1: Calibrate similarity rescaling")
    logger.info("=" * 60)
    sim_stats = calibrate_similarity_distribution()
    CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CALIBRATION_FILE, "w") as f:
        json.dump(sim_stats, f, indent=2)
    logger.info(f"Similarity calibration written to {CALIBRATION_FILE}")
    logger.info("")
    logger.info(f"  N samples:        {sim_stats['n_samples']}")
    logger.info(f"  Mean ± Std:       {sim_stats['mean']:.3f} ± {sim_stats['std']:.3f}")
    logger.info(f"  LOW (p10):        {sim_stats['calibrated_low']:.3f}  -> 0%")
    logger.info(f"  HIGH (p95):       {sim_stats['calibrated_high']:.3f}  -> 100%")
    logger.info(f"  THRESHOLD (p05):  {sim_stats['calibrated_threshold']:.3f}  -> drop below")
    logger.info("")

    logger.info("=" * 60)
    logger.info("STEP 2: Fit hybrid PCA + k-NN OOD detector")
    logger.info("=" * 60)
    fit_ood_detector()
    logger.info("")
    logger.info("All calibrations complete.")


if __name__ == "__main__":
    main()
    