"""Insights — operational dashboards driven by your real data and model.

The page is now organized into two tabs:
  1. Trends — the original 6 charts (KPIs, hour-of-day, amounts, importances,
     score distribution, threshold tradeoff, pattern frequency)
  2. Rule Generator — multi-agent pipeline that proposes production fraud rules
     from a filtered set of recent flagged transactions

Rule Generator gates on BYOK like the rest of the agentic surfaces.
"""

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Insights — FraudSentinel", layout="wide")

from src.dashboard import byok  # noqa: E402
from src.dashboard.agent_log import load_pattern_matches, load_recent_runs  # noqa: E402
from src.dashboard.components import (  # noqa: E402
    COLORS, _render_html, apply_theme, empty_state, icon, kpi_row,
    metric_card_with_sparkline, page_header, render_login_gate, render_top_bar,
    section_header, stage_indicator, status_pill, style_plotly,
)

apply_theme()
if not render_login_gate():
    st.stop()

render_top_bar()


@st.cache_data
def load_data() -> pd.DataFrame:
    from src.utils.config import SAMPLES_DIR
    path = SAMPLES_DIR / "demo_transactions.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def score_all(_df: pd.DataFrame) -> pd.DataFrame:
    if _df.empty:
        return _df
    from src.ml_models.inference import score_dataframe
    df = _df.copy()
    df["xgb_score"] = score_dataframe(df)
    return df


@st.cache_data
def load_feature_importances():
    """Load XGBoost feature importances grouped by category. Returns (df, error_message)."""
    try:
        from src.ml_models.inference import load_model
    except ImportError as e:
        return None, f"Cannot import load_model: {e}"

    try:
        loaded = load_model()
    except Exception as e:
        return None, f"load_model() failed: {e}"

    if loaded is None:
        return None, "load_model() returned None"

    model = None
    feature_names_from_tuple = None

    if isinstance(loaded, tuple):
        for item in loaded:
            if hasattr(item, "get_booster") or hasattr(item, "feature_importances_"):
                model = item
            elif isinstance(item, (list, tuple)) and len(item) > 0 and isinstance(item[0], str):
                feature_names_from_tuple = list(item)
    else:
        model = loaded

    if model is None:
        return None, "Could not find model in load_model() return"

    raw_importances = None
    names = None

    try:
        if hasattr(model, "feature_importances_"):
            raw_importances = model.feature_importances_
            if feature_names_from_tuple and len(feature_names_from_tuple) == len(raw_importances):
                names = feature_names_from_tuple
            elif hasattr(model, "feature_names_in_"):
                names = list(model.feature_names_in_)
            elif hasattr(model, "get_booster"):
                names = model.get_booster().feature_names or [f"f{i}" for i in range(len(raw_importances))]
        elif hasattr(model, "get_booster"):
            booster = model.get_booster()
            score = booster.get_score(importance_type="gain")
            names = list(score.keys())
            raw_importances = list(score.values())
    except Exception as e:
        return None, f"Reading importances failed: {e}"

    if raw_importances is None or names is None:
        return None, "Could not extract importances from model"

    def categorize(name: str) -> str:
        n = name.lower()
        if "txn_count" in n or "velocity" in n or "freq" in n:
            return "Velocity"
        if "amt" in n and ("zscore" in n or "ratio" in n or "mean" in n or "std" in n):
            return "Amount anomaly"
        if "transactionamt" in n or n == "amt":
            return "Amount"
        if n.startswith("card"):
            return "Card"
        if "emaildomain" in n or n.startswith("p_email") or n.startswith("r_email"):
            return "Email"
        if n.startswith("addr") or n == "dist1" or n == "dist2":
            return "Address / geo"
        if n.startswith("device") or n.startswith("id_"):
            return "Device / identity"
        if "transactiondt" in n or "hour" in n or "dow" in n or "day_of_week" in n:
            return "Time"
        if n.startswith("productcd") or n.startswith("product"):
            return "Channel"
        if n[:1] in ("v", "c", "d", "m") and (n[1:].isdigit() or (len(n) >= 2 and n[1:3].isdigit())):
            return "Anonymized (Vesta engineered)"
        return "Other"

    df_imp = pd.DataFrame({"feature": list(names), "importance": list(raw_importances)})
    df_imp = df_imp[df_imp["importance"] > 0]
    df_imp["category"] = df_imp["feature"].apply(categorize)

    grouped = df_imp.groupby("category")["importance"].sum().reset_index()
    grouped["pct"] = grouped["importance"] / grouped["importance"].sum() * 100
    grouped = grouped.sort_values("pct", ascending=True)

    top_by_cat = (
        df_imp.sort_values("importance", ascending=False)
        .groupby("category")
        .head(3)
        .groupby("category")["feature"]
        .apply(lambda s: ", ".join(s.tolist()))
        .reset_index()
        .rename(columns={"feature": "top_features"})
    )
    grouped = grouped.merge(top_by_cat, on="category", how="left")

    return grouped, None


df = load_data()
if df.empty:
    st.error("No transaction data available.")
    st.stop()
df = score_all(df)

has_truth = "isFraud" in df.columns


page_header(
    "Insights",
    "Patterns from your transaction data, model performance, and AI-generated production rules.",
)


tab_trends, tab_rules = st.tabs([
    "Trends & performance",
    "Rule Generator",
])


# ====================================================================
# TAB 1 — TRENDS (the original 6 charts, unchanged)
# ====================================================================
with tab_trends:
    # ------ KPIs ------
    total = len(df)
    flagged = int((df["xgb_score"] >= 0.6).sum())

    agent_runs = load_recent_runs(limit_hours=24 * 30)
    if not agent_runs.empty:
        run_totals = agent_runs.groupby("txn_id")["duration_ms"].sum()
        avg_lat = run_totals.mean()
        n_runs = len(run_totals)
    else:
        avg_lat = None
        n_runs = 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card_with_sparkline(
            label="Transactions", value=f"{total:,}",
            sparkline_values=[], delta="in current sample",
            sparkline_color=COLORS["accent"],
        )
    with col2:
        metric_card_with_sparkline(
            label="Flagged", value=f"{flagged:,}",
            sparkline_values=[], delta=f"{flagged/total*100:.1f}% of volume",
            sparkline_color=COLORS["high"],
        )
    with col3:
        if has_truth:
            confirmed = int(df["isFraud"].sum())
            catch = int(df[(df["xgb_score"] >= 0.6) & (df["isFraud"] == 1)].shape[0])
            catch_rate = (catch / confirmed * 100) if confirmed else 0
            metric_card_with_sparkline(
                label="Catch rate", value=f"{catch_rate:.1f}%",
                sparkline_values=[], delta=f"{catch} of {confirmed} caught",
                sparkline_color=COLORS["low"],
            )
        else:
            metric_card_with_sparkline(label="Catch rate", value="—", sparkline_values=[], sparkline_color=COLORS["low"])
    with col4:
        if avg_lat is not None:
            metric_card_with_sparkline(
                label="Avg agent latency", value=f"{avg_lat:.0f}ms",
                sparkline_values=[], delta=f"across {n_runs} run{'s' if n_runs != 1 else ''}",
                sparkline_color=COLORS["info"],
            )
        else:
            metric_card_with_sparkline(
                label="Avg agent latency", value="—",
                sparkline_values=[], delta="run an investigation to populate",
                sparkline_color=COLORS["info"],
            )

    # ------ CHART 1 — Hour of day ------
    section_header(
        "When fraud happens",
        "Fraud rate by hour of day. Higher bars mean a higher share of transactions in that hour are fraudulent.",
    )

    if not has_truth or "TransactionDT" not in df.columns:
        empty_state("Hour-of-day analysis needs `TransactionDT` and `isFraud` columns")
    else:
        df_h = df.copy()
        df_h["hour"] = ((df_h["TransactionDT"] % 86400) // 3600).astype(int)

        by_hour = df_h.groupby("hour").agg(
            total=("isFraud", "count"),
            fraud=("isFraud", "sum"),
        ).reset_index()
        by_hour["fraud_rate_pct"] = by_hour["fraud"] / by_hour["total"] * 100

        overall_rate = df_h["isFraud"].mean() * 100
        colors = [
            COLORS["critical"] if r >= overall_rate * 1.3
            else COLORS["high"] if r >= overall_rate
            else COLORS["accent"]
            for r in by_hour["fraud_rate_pct"]
        ]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=by_hour["hour"], y=by_hour["fraud_rate_pct"],
            marker=dict(color=colors, line=dict(width=0)),
            hovertemplate="<b>Hour %{x}:00</b><br>Fraud rate: %{y:.2f}%<br>%{customdata:,} transactions<extra></extra>",
            customdata=by_hour["total"],
        ))
        fig.add_hline(
            y=overall_rate, line_dash="dot", line_color=COLORS["text_muted"],
            annotation_text=f"avg {overall_rate:.2f}%", annotation_position="right",
            annotation_font_size=10, annotation_font_color=COLORS["text_muted"],
        )
        fig.update_layout(
            xaxis=dict(title=dict(text="Hour of day", font=dict(size=10)), dtick=2),
            yaxis=dict(title=dict(text="Fraud rate", font=dict(size=10)), ticksuffix="%"),
            showlegend=False, bargap=0.2,
        )
        style_plotly(fig, height=280)
        st.plotly_chart(fig, use_container_width=True)

    # ------ CHART 2 — Amounts ------
    section_header(
        "Amount distribution",
        "How transaction amounts differ between legitimate and fraudulent transactions (log scale).",
    )

    if not has_truth:
        empty_state("Amount comparison needs `isFraud` column")
    else:
        legit_amts = df[df["isFraud"] == 0]["TransactionAmt"]
        fraud_amts = df[df["isFraud"] == 1]["TransactionAmt"]

        log_min = np.log10(max(df["TransactionAmt"].min(), 0.1))
        log_max = np.log10(df["TransactionAmt"].max())
        bins = np.logspace(log_min, log_max, 30)

        legit_hist, _ = np.histogram(legit_amts, bins=bins)
        fraud_hist, _ = np.histogram(fraud_amts, bins=bins)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        legit_pct = legit_hist / legit_hist.sum() * 100 if legit_hist.sum() > 0 else legit_hist
        fraud_pct = fraud_hist / fraud_hist.sum() * 100 if fraud_hist.sum() > 0 else fraud_hist

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=bin_centers, y=legit_pct, name="Legitimate",
            marker=dict(color=COLORS["accent"], line=dict(width=0)), opacity=0.7,
            hovertemplate="<b>~$%{x:.0f}</b><br>%{y:.1f}% of legit<extra></extra>",
        ))
        fig2.add_trace(go.Bar(
            x=bin_centers, y=fraud_pct, name="Fraud",
            marker=dict(color=COLORS["critical"], line=dict(width=0)), opacity=0.7,
            hovertemplate="<b>~$%{x:.0f}</b><br>%{y:.1f}% of fraud<extra></extra>",
        ))
        fig2.update_layout(
            barmode="overlay",
            xaxis=dict(title=dict(text="Amount ($)", font=dict(size=10)), type="log"),
            yaxis=dict(title=dict(text="Share within group", font=dict(size=10)), ticksuffix="%"),
            legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
        )
        style_plotly(fig2, height=300)
        st.plotly_chart(fig2, use_container_width=True)

    # ------ CHART 3 — Feature importances ------
    section_header(
        "What drives the model",
        "Grouped by feature type. Shows where your model gets its signal — useful for spotting overreliance on any one source.",
    )

    imp_df, imp_err = load_feature_importances()
    if imp_df is None or len(imp_df) == 0:
        empty_state(
            "Feature importance unavailable",
            imp_err or "Could not load the trained model.",
        )
    else:
        CATEGORY_COLORS = {
            "Velocity": COLORS["high"], "Amount": COLORS["accent"],
            "Amount anomaly": COLORS["accent"], "Card": COLORS["info"],
            "Email": "#a78bfa", "Address / geo": "#84d4b8",
            "Device / identity": COLORS["medium"], "Time": "#9aafd4",
            "Channel": COLORS["low"], "Anonymized (Vesta engineered)": COLORS["text_dim"],
            "Other": COLORS["text_subtle"],
        }
        bar_colors = [CATEGORY_COLORS.get(c, COLORS["text_muted"]) for c in imp_df["category"]]

        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            y=imp_df["category"], x=imp_df["pct"], orientation="h",
            marker=dict(color=bar_colors, line=dict(width=0)),
            text=[f"{v:.1f}%" for v in imp_df["pct"]], textposition="outside",
            textfont=dict(color=COLORS["text"], size=11),
            customdata=imp_df["top_features"],
            hovertemplate="<b>%{y}</b><br>%{x:.1f}% of model decisions<br><span style='color:#9ba3b3'>Top features:</span> %{customdata}<extra></extra>",
        ))
        fig3.update_layout(
            xaxis=dict(title=dict(text="Share of model decision", font=dict(size=10)), ticksuffix="%"),
            yaxis=dict(title=None), showlegend=False,
        )
        style_plotly(fig3, height=max(280, 38 * len(imp_df) + 80))
        st.plotly_chart(fig3, use_container_width=True)

        top_cat = imp_df.nlargest(1, "pct").iloc[0]
        _render_html(
            f'<div style="color:{COLORS["text_muted"]};font-size:0.82rem;line-height:1.55;margin-top:0.6rem;background:{COLORS["surface_2"]};border:1px solid {COLORS["border"]};border-left:3px solid {COLORS["accent"]};border-radius:6px;padding:0.7rem 0.95rem;">'
            f'<strong style="color:{COLORS["text"]};">{top_cat["category"]}</strong> '
            f'features account for <strong style="color:{COLORS["accent"]};">{top_cat["pct"]:.1f}%</strong> of the model\'s decisions — '
            f'meaning the model leans most heavily on this signal type. Top contributors: '
            f'<span style="font-family:JetBrains Mono,monospace;font-size:0.8rem;">{top_cat["top_features"]}</span>.'
            f'</div>'
        )

    # ------ CHART 4 — Score distribution ------
    section_header(
        "Model confidence",
        "Distribution of risk scores across all transactions. A healthy model puts most legit transactions near 0 and most fraud near 1.",
    )

    if has_truth:
        legit_scores = df[df["isFraud"] == 0]["xgb_score"]
        fraud_scores = df[df["isFraud"] == 1]["xgb_score"]

        fig4 = go.Figure()
        fig4.add_trace(go.Histogram(
            x=legit_scores, name="Legitimate",
            marker=dict(color=COLORS["accent"], line=dict(width=0)),
            opacity=0.6, nbinsx=40,
            hovertemplate="Score %{x:.2f}<br>%{y:,} legit<extra></extra>",
        ))
        fig4.add_trace(go.Histogram(
            x=fraud_scores, name="Fraud",
            marker=dict(color=COLORS["critical"], line=dict(width=0)),
            opacity=0.7, nbinsx=40,
            hovertemplate="Score %{x:.2f}<br>%{y:,} fraud<extra></extra>",
        ))
        fig4.update_layout(
            barmode="overlay",
            xaxis=dict(title=dict(text="Risk score", font=dict(size=10)), tickformat=".0%"),
            yaxis=dict(title=dict(text="Transactions", font=dict(size=10)), type="log"),
            legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
        )
        style_plotly(fig4, height=300)
        st.plotly_chart(fig4, use_container_width=True)
    else:
        fig4 = go.Figure()
        fig4.add_trace(go.Histogram(
            x=df["xgb_score"], marker=dict(color=COLORS["accent"], line=dict(width=0)),
            opacity=0.85, nbinsx=40,
        ))
        fig4.update_layout(
            xaxis=dict(title=dict(text="Risk score", font=dict(size=10)), tickformat=".0%"),
            yaxis=dict(title=dict(text="Transactions", font=dict(size=10))),
        )
        style_plotly(fig4, height=300)
        st.plotly_chart(fig4, use_container_width=True)

    # ------ CHART 5 — Threshold tradeoff ------
    section_header(
        "Decision threshold",
        "Drag the slider to see how raising or lowering the cutoff trades catch rate against false positives.",
    )

    if not has_truth:
        empty_state(
            "Threshold analysis needs `isFraud` column",
            "Without ground truth labels, we can't compute catch rate or false positives.",
        )
    else:
        threshold = st.slider(
            "Threshold", min_value=0.05, max_value=0.95,
            value=0.6, step=0.05, format="%.2f",
            label_visibility="collapsed",
        )

        flagged_mask = df["xgb_score"] >= threshold
        fraud_mask = df["isFraud"] == 1
        legit_mask = df["isFraud"] == 0

        tp = int((flagged_mask & fraud_mask).sum())
        fn = int((~flagged_mask & fraud_mask).sum())
        fp = int((flagged_mask & legit_mask).sum())
        tn = int((~flagged_mask & legit_mask).sum())

        catch_rate = tp / max(tp + fn, 1) * 100
        fp_rate = fp / max(fp + tn, 1) * 100
        precision = tp / max(tp + fp, 1) * 100

        kpi_row([
            {"label": "At this threshold", "value": f"{threshold:.2f}", "sublabel": "score cutoff"},
            {"label": "Catch rate", "value": f"{catch_rate:.1f}%", "sublabel": f"{tp:,} of {tp+fn:,} fraud caught"},
            {"label": "False positive rate", "value": f"{fp_rate:.2f}%", "sublabel": f"{fp:,} legit flagged"},
            {"label": "Precision", "value": f"{precision:.1f}%", "sublabel": "of flagged are real fraud"},
        ])

        thresholds_grid = np.linspace(0.01, 0.99, 50)
        catch_curve = []
        fp_curve = []
        for t in thresholds_grid:
            fmask = df["xgb_score"] >= t
            c = (fmask & fraud_mask).sum() / max(fraud_mask.sum(), 1) * 100
            f = (fmask & legit_mask).sum() / max(legit_mask.sum(), 1) * 100
            catch_curve.append(c)
            fp_curve.append(f)

        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(
            x=fp_curve, y=catch_curve, mode="lines",
            line=dict(color=COLORS["accent"], width=2.5),
            fill="tozeroy", fillcolor="rgba(94,234,212,0.10)",
            hovertemplate="FPR %{x:.2f}%<br>Catch %{y:.1f}%<extra></extra>",
            name="All thresholds",
        ))
        fig5.add_trace(go.Scatter(
            x=[fp_rate], y=[catch_rate], mode="markers",
            marker=dict(size=14, color=COLORS["critical"], line=dict(color=COLORS["text"], width=2)),
            hovertemplate=f"<b>Current ({threshold:.2f})</b><br>FPR %{{x:.2f}}%<br>Catch %{{y:.1f}}%<extra></extra>",
            name="Current",
        ))
        fig5.update_layout(
            xaxis=dict(title=dict(text="False positive rate", font=dict(size=10)), ticksuffix="%"),
            yaxis=dict(title=dict(text="Catch rate", font=dict(size=10)), ticksuffix="%", range=[0, 105]),
            showlegend=False,
        )
        style_plotly(fig5, height=300)
        st.plotly_chart(fig5, use_container_width=True)

    # ------ CHART 6 — Pattern frequency ------
    section_header(
        "Top patterns this week",
        "Which fraud patterns the AI has surfaced most often during investigations.",
    )

    pattern_log = load_pattern_matches(limit_hours=24 * 7)

    if pattern_log.empty or len(pattern_log) < 3:
        empty_state(
            "Not enough investigation history yet",
            f"Run more investigations from the Investigate page. "
            f"({len(pattern_log)} match{'es' if len(pattern_log) != 1 else ''} logged so far — need at least 3 to draw the chart.)",
        )
    else:
        pattern_counts = (
            pattern_log["pattern"]
            .value_counts()
            .head(10)
            .reset_index()
        )
        pattern_counts.columns = ["pattern", "count"]
        pattern_counts["label"] = pattern_counts["pattern"].str.replace("_", " ").str.title()
        pattern_counts = pattern_counts.sort_values("count", ascending=True)

        PATTERN_COLORS_INSIGHTS = {
            "card_testing": COLORS["critical"], "geo_anomaly": COLORS["info"],
            "account_takeover": "#a78bfa", "velocity_attack": COLORS["high"],
            "synthetic_identity": COLORS["accent"], "bin_attack": COLORS["medium"],
            "friendly_fraud": COLORS["low"], "temporal_anomaly": "#f0a4d2",
            "email_risk": "#84d4b8", "subscription_probe": "#9aafd4",
            "device_takeover": "#7dd3fc", "credential_compromise": "#fbbf24",
            "engineered_anomaly": "#c084fc",
        }
        bar_colors = [PATTERN_COLORS_INSIGHTS.get(p, COLORS["text_muted"]) for p in pattern_counts["pattern"]]

        fig6 = go.Figure()
        fig6.add_trace(go.Bar(
            y=pattern_counts["label"], x=pattern_counts["count"], orientation="h",
            marker=dict(color=bar_colors, line=dict(width=0)),
            text=pattern_counts["count"], textposition="outside",
            textfont=dict(color=COLORS["text"], size=11),
            hovertemplate="<b>%{y}</b><br>%{x} matches<extra></extra>",
        ))
        fig6.update_layout(
            xaxis=dict(title=None, showticklabels=False),
            yaxis=dict(title=None), showlegend=False,
        )
        style_plotly(fig6, height=max(220, 30 * len(pattern_counts) + 60))
        st.plotly_chart(fig6, use_container_width=True)

        _render_html(
            f'<div style="color:{COLORS["text_dim"]};font-size:0.78rem;margin-top:0.6rem;">'
            f'Based on {len(pattern_log)} pattern matches across {pattern_log["txn_id"].nunique()} investigations in the last 7 days.'
            f'</div>'
        )


# ====================================================================
# TAB 2 — RULE GENERATOR
# ====================================================================
with tab_rules:
    section_header(
        "Rule Generator",
        "Multi-agent pipeline: a Planner reads aggregate stats from your selected fraud set, dispatches 4 parallel Workers (Velocity, Email, Device, Amount), and a Synthesizer ranks the proposed rules. Output: production-ready SQL + plain English.",
    )

    # ---- Controls row ----
    rg_c1, rg_c2, rg_c3 = st.columns([2, 2, 1])
    with rg_c1:
        risk_band = st.selectbox(
            "Risk band",
            options=["high_critical", "critical", "high", "all"],
            format_func=lambda x: {
                "high_critical": "High + Critical (score ≥ 0.6)",
                "critical": "Critical only (score ≥ 0.85)",
                "high": "High only (0.6 ≤ score < 0.85)",
                "all": "All transactions",
            }[x],
            help="Which model-flagged transactions to feed the pipeline. The aggregator also includes a baseline of unflagged transactions for fraud-vs-legit contrast.",
        )
    with rg_c2:
        date_range_days = st.selectbox(
            "Date range",
            options=[7, 14, 30, 60, 90],
            index=2,
            format_func=lambda x: f"Last {x} days",
            help="Date filter applied before risk-band filtering. Uses TransactionDT (IEEE-CIS seconds-from-epoch).",
        )
    with rg_c3:
        _render_html('<div style="height:28px;"></div>')
        run_rg_btn = st.button(
            "Generate rules",
            type="primary",
            use_container_width=True,
            key="run_rule_generator",
        )

    # ---- BYOK gate ----
    if not byok.has_api_key():
        byok.require_api_key(
            action_label="Generate production fraud rules",
            description="Runs a 6-call multi-agent pipeline (Planner → 4 parallel Workers → Synthesizer) over aggregate stats from your selected transactions. Key stays in your browser session only and is never saved.",
            estimated_cost="~$0.15 per generation",
        )
        st.stop()

    # ---- Cache key for results ----
    cache_key = f"rule_generator_result_{risk_band}_{date_range_days}"

    # ---- Run the pipeline ----
    if run_rg_btn:
        from src.agentic.rule_generator import RuleGenerator, filter_transactions

        # Filter the dataset
        with st.spinner("Filtering transactions and computing aggregates…"):
            filtered_df = filter_transactions(df, risk_band=risk_band, date_range_days=date_range_days)

        if len(filtered_df) < 5:
            st.error(
                f"Not enough transactions in the selected band/date range "
                f"({len(filtered_df)} found). Try a wider risk band or longer date range."
            )
            st.stop()

        # Show stage indicators
        stages_meta = [
            ("Aggregates", "Computing distributional stats from the filtered set"),
            ("Planner", "Deciding which workers to dispatch and writing focused briefs"),
            ("Workers", "4 parallel agents proposing velocity / email / device / amount rules"),
            ("Synthesis", "Ranking rules across workers, deduplicating, recommending deployment"),
        ]

        placeholder = st.empty()
        rendered_stages = {s_name: "running" for s_name, _ in stages_meta}

        def _render_pipeline_html():
            return "".join(
                f'<div style="display:flex;align-items:center;gap:14px;padding:0.7rem 0.95rem;border-bottom:1px solid {COLORS["border"]};">'
                f'<div style="min-width:170px;">{stage_indicator(s_name, status=rendered_stages[s_name])}</div>'
                f'<div style="color:{COLORS["text_muted"]};font-size:0.82rem;flex:1;">{s_desc}</div>'
                f'</div>'
                for s_name, s_desc in stages_meta
            )

        placeholder.markdown(
            f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};border-radius:8px;overflow:hidden;">{_render_pipeline_html()}</div>',
            unsafe_allow_html=True,
        )

        # Stream the pipeline
        api_key = byok.get_api_key()
        rg = RuleGenerator(api_key=api_key)

        stage_name_map = {
            "aggregates": "Aggregates",
            "planner": "Planner",
            "workers": "Workers",
            "synthesis": "Synthesis",
        }

        result_obj = None

        try:
            for event in rg.stream_generate(filtered_df, risk_band=risk_band):
                if event.get("type") == "stage":
                    stage_key = event.get("stage")
                    display_name = stage_name_map.get(stage_key, stage_key)
                    if event.get("status") == "done" and display_name in rendered_stages:
                        rendered_stages[display_name] = "done"
                        placeholder.markdown(
                            f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};border-radius:8px;overflow:hidden;">{_render_pipeline_html()}</div>',
                            unsafe_allow_html=True,
                        )
                elif event.get("type") == "complete":
                    result_obj = event.get("result")
                    if event.get("error"):
                        st.error(f"Generation error: {event['error']}")
                        st.stop()
        except Exception as e:
            st.error(f"Rule Generator failed: {type(e).__name__}: {e}")
            st.stop()

        if result_obj is None:
            st.error("Rule Generator returned no result.")
            st.stop()

        # Cache the result so reruns don't re-call the LLM
        st.session_state[cache_key] = result_obj

    # ---- Render cached result ----
    if cache_key not in st.session_state:
        _render_html(
            f'<div style="background:{COLORS["surface"]};border:1px dashed {COLORS["border_strong"]};border-radius:8px;padding:1.8rem;text-align:center;margin-top:1rem;">'
            f'<div style="color:{COLORS["text_muted"]};font-size:0.88rem;">Pick a risk band and date range, then click <strong style="color:{COLORS["text"]};">Generate rules</strong>.</div>'
            f'<div style="color:{COLORS["text_dim"]};font-size:0.76rem;margin-top:6px;">Pipeline runs ~5-7 seconds via parallel workers. Output: ranked production rules with SQL + plain English.</div>'
            f'</div>'
        )
        st.stop()

    result = st.session_state[cache_key]
    aggregates = result.aggregates
    plan = result.planner_output
    synthesis = result.synthesis

    # ---- Header summary ----
    section_header("Generation summary", "What the pipeline saw and what it produced")

    summary_html = (
        f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};border-radius:8px;padding:1rem 1.2rem;margin-bottom:1rem;">'
        f'<div style="display:flex;gap:24px;flex-wrap:wrap;">'
        f'<div><div style="color:{COLORS["text_dim"]};font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:4px;">Set size</div>'
        f'<div style="color:{COLORS["text"]};font-size:1.05rem;font-weight:700;">{aggregates.n_transactions:,} transactions</div></div>'
        f'<div><div style="color:{COLORS["text_dim"]};font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:4px;">Of which fraud</div>'
        f'<div style="color:{COLORS["critical"]};font-size:1.05rem;font-weight:700;">{aggregates.n_fraud:,} ({aggregates.fraud_rate*100:.1f}%)</div></div>'
        f'<div><div style="color:{COLORS["text_dim"]};font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:4px;">Rules proposed</div>'
        f'<div style="color:{COLORS["accent"]};font-size:1.05rem;font-weight:700;">{len(synthesis.ranked_rules)}</div></div>'
        f'<div><div style="color:{COLORS["text_dim"]};font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:4px;">Pipeline runtime</div>'
        f'<div style="color:{COLORS["text"]};font-size:1.05rem;font-weight:700;">{result.total_runtime_s:.1f}s</div></div>'
        f'</div>'
        f'</div>'
    )
    _render_html(summary_html)

    # ---- Strategy + coverage ----
    if plan.overall_strategy:
        _render_html(
            f'<div style="background:{COLORS["surface_2"]};border:1px solid {COLORS["border"]};border-left:3px solid {COLORS["accent"]};border-radius:6px;padding:0.85rem 1.1rem;margin-bottom:1rem;">'
            f'<div style="color:{COLORS["accent"]};font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:6px;">Planner strategy</div>'
            f'<div style="color:{COLORS["text_muted"]};font-size:0.86rem;line-height:1.55;">{plan.overall_strategy}</div>'
            f'</div>'
        )

    if synthesis.coverage_summary:
        _render_html(
            f'<div style="background:{COLORS["surface_2"]};border:1px solid {COLORS["border"]};border-left:3px solid {COLORS["info"]};border-radius:6px;padding:0.85rem 1.1rem;margin-bottom:1rem;">'
            f'<div style="color:{COLORS["info"]};font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:6px;">Coverage estimate</div>'
            f'<div style="color:{COLORS["text_muted"]};font-size:0.86rem;line-height:1.55;">{synthesis.coverage_summary}</div>'
            f'</div>'
        )

    if synthesis.deployment_recommendation:
        _render_html(
            f'<div style="background:{COLORS["surface_2"]};border:1px solid {COLORS["border"]};border-left:3px solid {COLORS["medium"]};border-radius:6px;padding:0.85rem 1.1rem;margin-bottom:1rem;">'
            f'<div style="color:{COLORS["medium"]};font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:6px;">Deployment recommendation</div>'
            f'<div style="color:{COLORS["text_muted"]};font-size:0.86rem;line-height:1.55;">{synthesis.deployment_recommendation}</div>'
            f'</div>'
        )

    # ---- Ranked rules ----
    section_header("Proposed rules", "Ranked by estimated lift across all workers")

    if not synthesis.ranked_rules:
        _render_html(
            f'<div style="background:{COLORS["surface"]};border:1px dashed {COLORS["border_strong"]};border-radius:8px;padding:1.4rem;text-align:center;color:{COLORS["text_muted"]};font-size:0.85rem;">'
            f'No rules were proposed. The aggregates may show too little signal in this band.'
            f'</div>'
        )
    else:
        SEVERITY_COLORS = {
            "block": COLORS["critical"],
            "review": COLORS["medium"],
            "monitor": COLORS["info"],
        }
        FAMILY_COLORS = {
            "velocity": COLORS["high"],
            "email": "#a78bfa",
            "device": "#7dd3fc",
            "amount": COLORS["accent"],
            "composite": "#fbbf24",
        }

        for i, rule in enumerate(synthesis.ranked_rules, 1):
            sev_color = SEVERITY_COLORS.get(rule.severity, COLORS["text_muted"])
            fam_color = FAMILY_COLORS.get(rule.feature_family, COLORS["text_muted"])

            evidence_html = ""
            if rule.evidence:
                evidence_html = (
                    f'<div style="margin-top:8px;color:{COLORS["text_dim"]};font-size:0.74rem;">'
                    + " · ".join(f'<span style="font-family:JetBrains Mono,monospace;">{e}</span>' for e in rule.evidence[:4])
                    + "</div>"
                )

            _render_html(
                f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};border-left:3px solid {sev_color};border-radius:8px;padding:1rem 1.2rem;margin-bottom:0.7rem;">'
                # Header row
                f'<div style="display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:8px;flex-wrap:wrap;">'
                f'<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">'
                f'<span style="color:{COLORS["text_dim"]};font-family:JetBrains Mono,monospace;font-size:0.78rem;">#{i}</span>'
                f'<span style="color:{COLORS["text"]};font-family:JetBrains Mono,monospace;font-size:0.86rem;font-weight:700;">{rule.rule_name}</span>'
                f'<span style="background:{fam_color};color:#0e1117;padding:2px 8px;border-radius:4px;font-size:0.66rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;">{rule.feature_family}</span>'
                f'<span style="background:{sev_color};color:#0e1117;padding:2px 8px;border-radius:4px;font-size:0.66rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;">{rule.severity}</span>'
                f'</div>'
                f'<div style="display:flex;gap:10px;font-size:0.75rem;color:{COLORS["text_dim"]};">'
                f'<span><strong style="color:{COLORS["low"]};">Catch:</strong> {rule.estimated_catch_rate}</span>'
                f'<span><strong style="color:{COLORS["high"]};">FPR:</strong> {rule.estimated_false_positive_rate}</span>'
                f'</div>'
                f'</div>'
                # Plain English
                f'<div style="color:{COLORS["text"]};font-size:0.92rem;line-height:1.55;margin-bottom:10px;">{rule.plain_english}</div>'
                # SQL
                f'<div style="background:{COLORS["surface_2"]};border:1px solid {COLORS["border"]};border-radius:4px;padding:8px 12px;margin-bottom:6px;">'
                f'<div style="color:{COLORS["text_dim"]};font-size:0.66rem;font-weight:600;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:4px;">SQL</div>'
                f'<div style="color:{COLORS["accent"]};font-family:JetBrains Mono,monospace;font-size:0.82rem;line-height:1.5;">{rule.rule_code_sql}</div>'
                f'</div>'
                # Pseudo
                f'<div style="background:{COLORS["surface_2"]};border:1px solid {COLORS["border"]};border-radius:4px;padding:8px 12px;margin-bottom:6px;">'
                f'<div style="color:{COLORS["text_dim"]};font-size:0.66rem;font-weight:600;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:4px;">Pseudo-code</div>'
                f'<div style="color:{COLORS["text_muted"]};font-family:JetBrains Mono,monospace;font-size:0.82rem;line-height:1.5;">{rule.rule_code_pseudo}</div>'
                f'</div>'
                # Rationale
                f'<div style="color:{COLORS["text_muted"]};font-size:0.78rem;line-height:1.55;margin-top:8px;font-style:italic;">'
                f'<strong style="color:{COLORS["text"]};font-style:normal;">Rationale:</strong> {rule.rationale}'
                f'</div>'
                f'{evidence_html}'
                f'</div>'
            )

    # ---- Worker breakdown (collapsible detail) ----
    with st.expander("Worker breakdown — what each agent contributed", expanded=False):
        for w_name, w_out in result.worker_outputs.items():
            w_color = FAMILY_COLORS.get(w_name, COLORS["text_muted"]) if (FAMILY_COLORS := {
                "velocity": COLORS["high"], "email": "#a78bfa",
                "device": "#7dd3fc", "amount": COLORS["accent"],
            }) else COLORS["text_muted"]

            error_pill = ""
            if w_out.error:
                error_pill = (
                    f'<span style="background:{COLORS["critical"]};color:#0e1117;padding:1px 7px;border-radius:3px;font-size:0.62rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;margin-left:8px;">ERROR</span>'
                )

            _render_html(
                f'<div style="background:{COLORS["surface_2"]};border:1px solid {COLORS["border"]};border-left:3px solid {w_color};border-radius:6px;padding:0.85rem 1.1rem;margin-bottom:0.6rem;">'
                f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">'
                f'<span style="color:{w_color};font-size:0.72rem;font-weight:600;text-transform:uppercase;letter-spacing:0.07em;">{w_name} worker</span>'
                f'<span style="color:{COLORS["text_dim"]};font-size:0.72rem;font-family:JetBrains Mono,monospace;">{w_out.runtime_ms:.0f}ms · {len(w_out.proposed_rules)} rule{"s" if len(w_out.proposed_rules) != 1 else ""}</span>'
                f'{error_pill}'
                f'</div>'
                + (f'<div style="color:{COLORS["text"]};font-size:0.84rem;line-height:1.5;font-weight:600;margin-bottom:4px;">{w_out.key_finding}</div>' if w_out.key_finding else "")
                + (f'<div style="color:{COLORS["text_muted"]};font-size:0.8rem;line-height:1.55;">{w_out.summary}</div>' if w_out.summary else "")
                + (f'<div style="color:{COLORS["critical"]};font-size:0.78rem;margin-top:6px;font-family:JetBrains Mono,monospace;">{w_out.error}</div>' if w_out.error else "")
                + f'</div>'
            )

    # ---- Aggregate stats (collapsible detail) ----
    with st.expander("Aggregate stats — the data the workers saw", expanded=False):
        st.json(aggregates.to_dict())
        