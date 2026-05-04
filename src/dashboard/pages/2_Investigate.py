"""Investigate — review a flagged transaction with AI-assisted decisioning.

Five-tab layout: Summary | Score Attribution | Evidence | Patterns | Checklist

The Checklist tab is ADAPTIVE:
  - When the Pattern agent reports a Strong fit OR Partial fit → uses
    pattern-grounded coach with paired plain-English label + technical
    receipt rendering, plus a five-tier indicator-overlap verdict
  - When the Pattern agent reports No fit → uses SHAP-grounded coach with
    HYBRID STREAMING:
        1. Rule-based baseline renders instantly (~50ms)
        2. LLM enrichment auto-triggers in background (~3s)
        3. Generic checks smoothly replaced by transaction-specific AI questions

ROUTING DETECTION:
  Pattern agent emits one of three lead sentence families:
    "[This transaction] matches the [X] pattern."             → Strong fit  → Branch A
    "[This transaction] strongly/closely matches the X..."    → Strong fit  → Branch A
    "[This transaction] partially matches the X pattern."     → Partial fit → Branch A
    "The retrieved patterns are the closest semantic
     matches but none actually fits."                         → No fit      → Branch B

  The router uses the substring "matches the" (catching all variants like
  "matches", "strongly matches", "closely matches") combined with negative
  signals ("none actually fits", "no pattern from the library") to determine
  branching, rather than requiring an exact phrase match.

BYOK gating:
  - "Run investigation" button gates inline if no Anthropic API key is connected
  - Free tabs (Score Attribution, Patterns) work without a key once an
    investigation has run; if no investigation has run, they show empty
"""

import logging
import re as _re
import time as _time

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Investigate — FraudSentinel", layout="wide")

from dotenv import load_dotenv  # noqa: E402

from src.dashboard import byok  # noqa: E402
from src.dashboard import byok  # noqa: E402  (existing line — for reference)
from src.agentic.investigation_cache import (  # noqa: E402
    get_cached_investigation,
    store_investigation,
)
from src.dashboard.agent_log import time_stage  # noqa: E402
from src.dashboard.components import (  # noqa: E402
    COLORS, _render_html, apply_theme, detail_grid, icon,
    page_header, render_login_gate, render_top_bar, risk_indicator,
    section_header, stage_indicator, status_pill,
)
from src.dashboard.pattern_match import find_similar_patterns  # noqa: E402

load_dotenv()
apply_theme()
if not render_login_gate():
    st.stop()

render_top_bar()

logger = logging.getLogger(__name__)


def _escape_dollars(text: str) -> str:
    if not text:
        return text
    return _re.sub(r"(?<!\\)\$", r"\\$", text)


# ====================================================================
# Data loading
# ====================================================================
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


df = load_data()
if df.empty:
    st.error("No transaction data available.")
    st.stop()
df = score_all(df)


page_header(
    "Investigate",
    "Review a flagged transaction. Multi-agent AI gathers evidence, attributes the score, matches patterns, and explains the decision.",
)


# ====================================================================
# Transaction picker
# ====================================================================
selected_id = st.session_state.get("selected_transaction_id")

c1, c2 = st.columns([3, 1])
with c1:
    txn_options = df.nlargest(50, "xgb_score")["TransactionID"].astype(str).tolist()
    default_idx = 0
    if selected_id and selected_id in txn_options:
        default_idx = txn_options.index(selected_id)
    pick = st.selectbox(
        "Choose a transaction to investigate",
        options=txn_options,
        index=default_idx,
        format_func=lambda x: f"Txn {x}",
    )
with c2:
    _render_html('<div style="height:28px;"></div>')
    run_btn = st.button("Run investigation", use_container_width=True, type="primary")


row = df[df["TransactionID"].astype(str) == pick].iloc[0]
score = float(row.get("xgb_score", 0))
amt = float(row.get("TransactionAmt", 0))
gt = int(row.get("isFraud", 0)) if "isFraud" in row else None


# ====================================================================
# Persistent header
# ====================================================================
left, right = st.columns([2, 1])

with left:
    section_header("Transaction summary")
    detail_grid([
        {"label": "Transaction", "value": str(row.get("TransactionID", "")), "mono": True},
        {"label": "Amount", "value": f"${amt:,.2f}"},
        {"label": "Channel", "value": str(row.get("ProductCD", "?"))},
        {"label": "Card BIN", "value": str(row.get("card1", "")), "mono": True},
        {"label": "Card type", "value": f"{row.get('card4', '?')} • {row.get('card6', '?')}"},
        {"label": "24h velocity", "value": f"{int(row.get('card1_txn_count_24h', 0))} transactions"},
        {"label": "Amount anomaly", "value": f"{float(row.get('card1_amt_zscore', 0)):+.2f}σ from card avg"},
    ])

with right:
    section_header("Risk score")
    risk_indicator(score)
    _render_html(
        f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};border-radius:8px;padding:0.9rem 1.1rem;margin-top:0.6rem;">'
        f'<div style="color:{COLORS["text_muted"]};font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:6px;">Model output</div>'
        f'<div style="color:{COLORS["text"]};font-size:1.5rem;font-weight:700;font-family:Inter,sans-serif;line-height:1.1;">{score*100:.2f}%</div>'
        f'<div style="color:{COLORS["text_dim"]};font-size:0.74rem;margin-top:5px;">Probability of fraud</div>'
        f'</div>'
    )
    if gt is not None:
        section_header("Ground truth")
        gt_label = "Confirmed fraud" if gt == 1 else "Legitimate"
        gt_color = COLORS["critical"] if gt == 1 else COLORS["low"]
        gt_bg = COLORS["critical_bg"] if gt == 1 else COLORS["low_bg"]
        _render_html(
            f'<div style="background:{gt_bg};border:1px solid {gt_color};border-radius:8px;padding:0.7rem 0.95rem;color:{gt_color};font-size:0.86rem;font-weight:600;display:flex;align-items:center;gap:8px;">'
            f'{icon("shield-check" if gt == 0 else "alert-triangle", size=14, color=gt_color)}<span>{gt_label}</span></div>'
        )


# ====================================================================
# Investigation orchestration
# ====================================================================
investigation_key = f"investigation_{pick}"
already_done = investigation_key in st.session_state

if not run_btn and not already_done:
    section_header("AI investigation")

    if not byok.has_api_key():
        byok.require_api_key(
            action_label="Run multi-agent investigation",
            description="Runs 4 specialized Claude agents (Triage, Investigator with tool use, Pattern, Report) over deterministic SHAP and LSTM attribution. Key stays in your browser session only and is never saved.",
            estimated_cost="~$0.05 per investigation",
        )
    else:
        _render_html(
            f'<div style="background:{COLORS["surface"]};border:1px dashed {COLORS["border_strong"]};border-radius:8px;padding:1.8rem;text-align:center;">'
            f'<div style="color:{COLORS["text_muted"]};font-size:0.88rem;">Click <strong style="color:{COLORS["text"]};">Run investigation</strong> to start the AI workflow.</div>'
            f'<div style="color:{COLORS["text_dim"]};font-size:0.76rem;margin-top:6px;">Triage → Attribution → Evidence → Pattern match → Analyst explanation</div>'
            f'</div>'
        )
    st.stop()


def _compute_real_lstm_signal(_row, full_df: pd.DataFrame) -> tuple[float, bool]:
    try:
        from src.dl_models.inference import score_sequence

        card_id = _row.get("card1")
        txn_id = _row.get("TransactionID")
        if card_id is None or full_df.empty:
            return 0.5, False

        history = full_df[full_df["card1"] == card_id]
        if "TransactionDT" in history.columns:
            history = history.sort_values("TransactionDT")

        if txn_id is not None and txn_id in history["TransactionID"].values:
            idx = history.index[history["TransactionID"] == txn_id][-1]
            pos = history.index.get_loc(idx)
            window = history.iloc[max(0, pos - 50): pos + 1]
        else:
            window = history.tail(50)

        if window.empty:
            return 0.5, False

        result = score_sequence(window)
        return (
            float(result.get("normalized_score", 0.5)),
            bool(result.get("is_anomaly", False)),
        )
    except Exception as e:
        logger.warning(f"LSTM scoring failed: {e}")
        return 0.5, False


def _run_agent_for_row(_row, _xgb_score: float, _full_df: pd.DataFrame) -> dict:
    from src.agentic.orchestrator import FraudInvestigator

    api_key = byok.get_api_key()
    if not api_key:
        return {
            "error": "Anthropic API key required. Connect your key in Settings to run investigations.",
            "triage": None, "attribution": None, "investigator": None,
            "pattern": None, "report": None,
        }

    txn = _row.to_dict()
    for k in ("isFraud",):
        txn.pop(k, None)
    for k, v in list(txn.items()):
        if hasattr(v, "isoformat"):
            txn[k] = v.isoformat()

    lstm_normalized, lstm_is_anomaly = _compute_real_lstm_signal(_row, _full_df)

    investigator = FraudInvestigator(api_key=api_key)
    stages_done = {
        "triage": None, "attribution": None, "investigator": None,
        "pattern": None, "report": None,
    }

    try:
        for event in investigator.stream_investigate(
            transaction=txn,
            xgb_score=float(_xgb_score),
            lstm_score=lstm_normalized,
            lstm_anomaly=lstm_is_anomaly,
        ):
            if event.get("type") == "stage" and event.get("status") == "done":
                stage = event["stage"]
                stages_done[stage] = event.get("data")
            elif event.get("type") == "complete":
                ctx = event.get("context")
                if ctx is not None:
                    stages_done["triage"] = ctx.triage_result
                    stages_done["attribution"] = {
                        "xgb": ctx.xgb_attribution,
                        "lstm": ctx.lstm_attribution,
                    }
                    stages_done["investigator"] = {
                        "findings": ctx.investigator_findings,
                        "tool_calls": ctx.investigator_tool_calls,
                    }
                    stages_done["pattern"] = {
                        "matches": ctx.pattern_matches,
                        "analysis": ctx.pattern_analysis,
                    }
                    stages_done["report"] = ctx.final_report
        return {
            "error": None,
            "triage": stages_done["triage"],
            "attribution": stages_done["attribution"],
            "investigator": stages_done["investigator"],
            "pattern": stages_done["pattern"],
            "report": stages_done["report"],
        }
    except Exception as e:
        return {
            "error": f"Investigation failed: {type(e).__name__}: {e}",
            "triage": stages_done["triage"],
            "attribution": stages_done["attribution"],
            "investigator": stages_done["investigator"],
            "pattern": stages_done["pattern"],
            "report": stages_done["report"],
        }


if run_btn or not already_done:
    section_header("AI investigation")

    stages_meta = [
        ("Triage", "Routing this case to the right workflow"),
        ("Attribution", "Computing SHAP + LSTM timestep attributions"),
        ("Evidence", "Pulling card history, merchant profile, velocity signals via tools"),
        ("Pattern match", "Comparing to similar historical fraud cases"),
        ("Analyst explanation", "Synthesizing the analyst-facing decision summary"),
    ]

    placeholder = st.empty()
    pipeline_html = "".join(
        f'<div style="display:flex;align-items:center;gap:14px;padding:0.7rem 0.95rem;border-bottom:1px solid {COLORS["border"]};">'
        f'<div style="min-width:170px;">{stage_indicator(s_name, status="running")}</div>'
        f'<div style="color:{COLORS["text_muted"]};font-size:0.82rem;flex:1;">{s_desc}</div>'
        f'</div>'
        for s_name, s_desc in stages_meta
    )
    placeholder.markdown(
        f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};border-radius:8px;overflow:hidden;">{pipeline_html}</div>',
        unsafe_allow_html=True,
    )

    t0 = _time.time()

    # Check cache first — keyed by (transaction_id, feature_hash, model_version).
    # A hit returns the previous output byte-for-byte, eliminating ~30-60s
    # of LLM latency for re-views.
    txn_dict = row.to_dict()
    for k, v in list(txn_dict.items()):
        if hasattr(v, "isoformat"):
            txn_dict[k] = v.isoformat()

    from src.utils.config import (
        TRIAGE_MODEL, PATTERN_MODEL, INVESTIGATOR_MODEL, REPORT_MODEL,
    )

    cached = get_cached_investigation(
        txn_dict,
        triage_model=TRIAGE_MODEL,
        pattern_model=PATTERN_MODEL,
        investigator_model=INVESTIGATOR_MODEL,
        report_model=REPORT_MODEL,
    )

    if cached is not None:
        result = cached
        result["from_cache"] = True
    else:
        with time_stage(txn_id=str(pick), stage="full_investigation"):
            result = _run_agent_for_row(row, score, df)
        result["from_cache"] = False
        # Store in cache for future re-views
        if not result.get("error") or result.get("report"):
            store_investigation(
                txn_dict,
                result,
                triage_model=TRIAGE_MODEL,
                pattern_model=PATTERN_MODEL,
                investigator_model=INVESTIGATOR_MODEL,
                report_model=REPORT_MODEL,
            )

    duration_s = _time.time() - t0
    result["duration_s"] = duration_s
    st.session_state[investigation_key] = result

    final_html = "".join(
        f'<div style="display:flex;align-items:center;gap:14px;padding:0.7rem 0.95rem;border-bottom:1px solid {COLORS["border"]};">'
        f'<div style="min-width:170px;">{stage_indicator(s_name, status="done")}</div>'
        f'<div style="color:{COLORS["text_muted"]};font-size:0.82rem;flex:1;">{s_desc}</div>'
        f'</div>'
        for s_name, s_desc in stages_meta
    )
    placeholder.markdown(
        f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};border-radius:8px;overflow:hidden;">{final_html}</div>',
        unsafe_allow_html=True,
    )

result = st.session_state[investigation_key]


if result.get("error") and not result.get("report"):
    st.error(result["error"])
    st.info(
        "The static recommendation below is computed from the risk score directly "
        "as a fallback while the AI agent is unavailable."
    )


# ====================================================================
# Pre-compute pattern candidates
# ====================================================================
PATTERN_COLORS_INVESTIGATE = {
    "card_testing": COLORS["critical"],
    "geo_anomaly": COLORS["info"],
    "account_takeover": "#a78bfa",
    "velocity_attack": COLORS["high"],
    "synthetic_identity": COLORS["accent"],
    "bin_attack": COLORS["medium"],
    "friendly_fraud": COLORS["low"],
    "temporal_anomaly": "#f0a4d2",
    "email_risk": "#84d4b8",
    "subscription_probe": "#9aafd4",
    "device_takeover": "#7dd3fc",
    "credential_compromise": "#fbbf24",
    "engineered_anomaly": "#c084fc",
}


def _matches_are_usable(matches) -> bool:
    if not matches or not isinstance(matches, list):
        return False
    first = matches[0]
    if not isinstance(first, dict):
        return False
    if first.get("ood_rejected") or first.get("error"):
        return False
    return bool(first.get("title") or first.get("pattern"))


agent_pattern_matches = (result.get("pattern") or {}).get("matches") or []
similar_cache_key = f"similar_{pick}"

if _matches_are_usable(agent_pattern_matches):
    similar = agent_pattern_matches
elif similar_cache_key in st.session_state:
    similar = st.session_state[similar_cache_key]
else:
    with st.spinner("Searching pattern library…"):
        similar = find_similar_patterns(row, top_k=3, log_for_txn_id=str(pick))
    st.session_state[similar_cache_key] = similar


# ====================================================================
# TABS
# ====================================================================
tab_summary, tab_attribution, tab_evidence, tab_patterns, tab_checklist = st.tabs([
    "Summary",
    "Score Attribution",
    "Evidence",
    "Patterns",
    "Checklist",
])


# ====================================================================
# TAB 1 — Summary
# ====================================================================
with tab_summary:
    triage = result.get("triage") or {}
    if triage:
        section_header("Triage verdict", "Quick risk routing decision from the agent")
        risk_level = str(triage.get("risk", "?")).upper()
        risk_color_map = {
            "CRITICAL": COLORS["critical"], "HIGH": COLORS["high"],
            "MEDIUM": COLORS["medium"], "LOW": COLORS["low"],
        }
        risk_color = risk_color_map.get(risk_level, COLORS["text_muted"])
        investigate_flag = triage.get("investigate", True)
        reason = triage.get("reason", "")

        _render_html(
            f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};border-left:3px solid {risk_color};'
            f'border-radius:8px;padding:1rem 1.2rem;margin-bottom:1rem;">'
            f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">'
            f'<span style="background:{risk_color};color:#0e1117;padding:3px 10px;border-radius:4px;font-size:0.7rem;font-weight:700;'
            f'text-transform:uppercase;letter-spacing:0.07em;">{risk_level}</span>'
            f'<span style="color:{COLORS["text_dim"]};font-size:0.78rem;">'
            f'{"Routed for full investigation" if investigate_flag else "Closed at triage"}</span>'
            f'</div>'
            f'<div style="color:{COLORS["text_muted"]};font-size:0.86rem;line-height:1.55;">{reason}</div>'
            f'</div>'
        )

    report_text = result.get("report")
    if report_text:
        section_header(
            "Analyst explanation",
            "The headline synthesis — what to know in 30 seconds",
        )
        with st.container():
            st.markdown(_escape_dollars(report_text))
        duration_s = result.get("duration_s")
        from_cache = result.get("from_cache", False)
        if duration_s:
            if from_cache:
                footer_text = (
                    f"Loaded from cache in {duration_s*1000:.0f}ms — "
                    f"this transaction was investigated before with the same model versions."
                )
            else:
                footer_text = (
                    f"Generated in {duration_s:.1f}s by 4 LLM agents + 2 deterministic attribution stages "
                    f"(Haiku for triage/pattern, Sonnet for investigator/report)."
                )
            _render_html(
                f'<div style="color:{COLORS["text_dim"]};font-size:0.74rem;margin-top:0.4rem;margin-bottom:1rem;">'
                f'{footer_text}'
                f'</div>'
            )

    section_header("Recommendation", "Quick verdict from the risk score")

    if score >= 0.85:
        rec, rec_key, body = "Decline", "critical", "High fraud probability with corroborating velocity and amount anomalies. Recommend declining and adding the card to watch list."
    elif score >= 0.6:
        rec, rec_key, body = "Hold for review", "high", "Elevated risk. Recommend a manual review before settling. Issue a step-up authentication if the customer is reachable."
    elif score >= 0.3:
        rec, rec_key, body = "Approve with monitoring", "medium", "Risk indicators are mild. Recommend approving and tagging the card for 24-hour monitoring."
    else:
        rec, rec_key, body = "Approve", "low", "Low risk. Approve and continue."

    rec_color = COLORS[rec_key]
    rec_bg = COLORS[f"{rec_key}_bg"]
    _render_html(
        f'<div style="background:{rec_bg};border:1px solid {rec_color};border-radius:8px;padding:1.1rem 1.3rem;">'
        f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">'
        f'{status_pill(rec.upper(), rec_key)}'
        f'<span style="color:{COLORS["text"]};font-size:0.95rem;font-weight:600;">{rec}</span>'
        f'</div>'
        f'<div style="color:{COLORS["text_muted"]};font-size:0.86rem;line-height:1.55;">{body}</div>'
        f'</div>'
    )

    _render_html('<div style="height:1.2rem;"></div>')
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        st.button("Approve", use_container_width=True, key="dec_approve")
    with b2:
        st.button("Hold for review", use_container_width=True, key="dec_hold")
    with b3:
        st.button("Decline", use_container_width=True, key="dec_decline")
    with b4:
        st.button("Escalate", use_container_width=True, key="dec_escalate")


# ====================================================================
# TAB 2 — Score Attribution
# ====================================================================
with tab_attribution:
    attr = result.get("attribution") or {}
    xgb_attr = attr.get("xgb") or {}
    lstm_attr = attr.get("lstm") or {}

    if not xgb_attr and not lstm_attr:
        _render_html(
            f'<div style="background:{COLORS["surface"]};border:1px dashed {COLORS["border_strong"]};border-radius:8px;padding:1.4rem;text-align:center;color:{COLORS["text_muted"]};font-size:0.85rem;">'
            f'No attribution data available for this investigation.'
            f'</div>'
        )

    if xgb_attr and not xgb_attr.get("error") and xgb_attr.get("top_drivers"):
        section_header(
            "XGBoost feature contributions",
            "Which features actually drove the fraud probability — computed via SHAP TreeExplainer",
        )

        proba = xgb_attr.get("predicted_proba") or 0
        signal_balance = xgb_attr.get("signal_balance", "unknown")
        balance_label = {
            "fraud_signals_dominate": ("Fraud signals dominate", COLORS["critical"]),
            "legit_signals_dominate": ("Legitimate signals dominate", COLORS["low"]),
            "mixed": ("Mixed signals", COLORS["medium"]),
        }.get(signal_balance, (signal_balance.replace("_", " ").title(), COLORS["text_muted"]))

        _render_html(
            f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};border-radius:8px;padding:1rem 1.2rem;margin-bottom:0.8rem;">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;gap:14px;margin-bottom:14px;">'
            f'<div>'
            f'<div style="color:{COLORS["text_dim"]};font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:4px;">XGBoost SHAP attribution</div>'
            f'<div style="color:{COLORS["text"]};font-size:0.95rem;font-weight:600;">{proba*100:.2f}% fraud probability</div>'
            f'</div>'
            f'<span style="background:{balance_label[1]};color:#0e1117;padding:3px 10px;border-radius:4px;font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:0.07em;">{balance_label[0]}</span>'
            f'</div>'
        )

        drivers = xgb_attr.get("top_drivers", [])
        max_abs = max((abs(d["shap"]) for d in drivers), default=1.0) or 1.0

        for d in drivers:
            shap_val = d["shap"]
            bar_width_pct = abs(shap_val) / max_abs * 100
            bar_color = COLORS["critical"] if shap_val > 0 else COLORS["low"]
            sign = "+" if shap_val > 0 else "−"
            family_label = d["family"].replace("_", " ")
            _render_html(
                f'<div style="margin-bottom:10px;">'
                f'<div style="display:flex;justify-content:space-between;font-size:0.8rem;margin-bottom:4px;gap:14px;">'
                f'<div style="color:{COLORS["text"]};flex:1;overflow:hidden;text-overflow:ellipsis;">'
                f'<span style="font-weight:600;">{d["label"]}</span>'
                f'<span style="color:{COLORS["text_dim"]};font-family:JetBrains Mono,monospace;font-size:0.74rem;margin-left:8px;">= {d["value"]}</span>'
                f'<span style="color:{COLORS["text_dim"]};font-size:0.7rem;margin-left:8px;text-transform:uppercase;letter-spacing:0.06em;">[{family_label}]</span>'
                f'</div>'
                f'<div style="color:{bar_color};font-weight:700;font-family:JetBrains Mono,monospace;flex-shrink:0;">{sign}{abs(shap_val):.3f}</div>'
                f'</div>'
                f'<div style="background:{COLORS["surface_2"]};height:6px;border-radius:3px;overflow:hidden;">'
                f'<div style="background:{bar_color};height:100%;width:{bar_width_pct:.1f}%;"></div>'
                f'</div>'
                f'</div>'
            )

        families = xgb_attr.get("by_family", {})
        if families:
            family_pills = " ".join(
                f'<span style="background:{COLORS["surface_2"]};border:1px solid {COLORS["border"]};color:{COLORS["text_muted"]};'
                f'padding:3px 9px;border-radius:4px;font-size:0.74rem;font-family:JetBrains Mono,monospace;margin-right:5px;'
                f'display:inline-block;margin-bottom:4px;">'
                f'{fam.replace("_", " ")}: {"+" if val > 0 else "−"}{abs(val):.2f}'
                f'</span>'
                for fam, val in list(families.items())[:6]
            )
            _render_html(
                f'<div style="margin-top:14px;padding-top:12px;border-top:1px solid {COLORS["border"]};">'
                f'<div style="color:{COLORS["text_dim"]};font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:7px;">Total contribution by feature family</div>'
                f'{family_pills}'
                f'</div>'
                f'</div>'
            )
        else:
            _render_html('</div>')

    elif xgb_attr and xgb_attr.get("error"):
        section_header("XGBoost feature contributions")
        _render_html(
            f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};border-left:3px solid {COLORS["medium"]};border-radius:8px;padding:0.9rem 1.1rem;margin-bottom:0.8rem;color:{COLORS["text_muted"]};font-size:0.82rem;">'
            f'XGBoost SHAP attribution unavailable: {xgb_attr["error"]}'
            f'</div>'
        )

    if lstm_attr and not lstm_attr.get("error") and lstm_attr.get("top_steps"):
        section_header(
            "Sequence model — recent activity check",
            "What the LSTM autoencoder sees in the card's recent transaction history",
        )

        verdict = lstm_attr.get("verdict", "unknown")
        is_current_anomaly = verdict == "anomaly_at_current"
        is_earlier_anomaly = verdict == "anomaly_earlier_in_window"

        if is_current_anomaly:
            verdict_headline = "This transaction breaks the card's recent pattern"
            verdict_body = (
                "The autoencoder model — which has learned what normal activity looks "
                "like for this card — flagged THIS transaction as the most unusual one in "
                "its recent history. The XGBoost score and the sequence model agree."
            )
            verdict_color = COLORS["critical"]
            verdict_bg = COLORS["critical_bg"]
            verdict_icon = "alert-triangle"
        elif is_earlier_anomaly:
            verdict_headline = "Something unusual happened earlier on this card"
            verdict_body = (
                "The current transaction looks normal compared to recent activity, but "
                "the model flagged an EARLIER transaction in the sequence as anomalous. "
                "The card may have been compromised before this charge — review the recent history."
            )
            verdict_color = COLORS["medium"]
            verdict_bg = COLORS["medium_bg"]
            verdict_icon = "clock"
        else:
            verdict_headline = "The card's recent activity looks normal"
            verdict_body = (
                "The autoencoder model reconstructs the recent transaction sequence "
                "without trouble. If XGBoost flagged this transaction, the signal is "
                "coming from features the sequence model doesn't see (device, identity, etc.)."
            )
            verdict_color = COLORS["low"]
            verdict_bg = COLORS["low_bg"]
            verdict_icon = "shield-check"

        anomaly_score = lstm_attr.get("anomaly_score", 0)
        threshold = lstm_attr.get("threshold", 0)
        seq_len = lstm_attr.get("sequence_length", 0)
        current_rank = lstm_attr.get("current_step_rank", "?")

        if threshold > 0:
            multiple = anomaly_score / threshold
            if multiple >= 1:
                multiple_str = f"{multiple:.1f}× above the alert threshold"
                multiple_color = COLORS["critical"]
            else:
                multiple_str = f"{multiple:.1f}× of the alert threshold (under)"
                multiple_color = COLORS["low"]
        else:
            multiple_str = ""
            multiple_color = COLORS["text_muted"]

        _render_html(
            f'<div style="background:{verdict_bg};border:1px solid {verdict_color};border-radius:8px;padding:1rem 1.2rem;margin-bottom:0.6rem;">'
            f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">'
            f'{icon(verdict_icon, size=18, color=verdict_color)}'
            f'<div style="color:{verdict_color};font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.07em;">Sequence model (LSTM autoencoder)</div>'
            f'</div>'
            f'<div style="color:{COLORS["text"]};font-size:1.05rem;font-weight:700;line-height:1.35;margin-bottom:6px;">{verdict_headline}</div>'
            f'<div style="color:{COLORS["text_muted"]};font-size:0.86rem;line-height:1.55;">{verdict_body}</div>'
            f'</div>'
        )

        _render_html(
            f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};border-radius:8px;padding:0.85rem 1.1rem;margin-bottom:0.6rem;display:flex;gap:24px;">'
            f'<div style="flex:1;">'
            f'<div style="color:{COLORS["text_dim"]};font-size:0.68rem;font-weight:600;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:4px;">Window analyzed</div>'
            f'<div style="color:{COLORS["text"]};font-size:0.92rem;font-weight:600;">Last {seq_len} transactions</div>'
            f'</div>'
            f'<div style="flex:1;">'
            f'<div style="color:{COLORS["text_dim"]};font-size:0.68rem;font-weight:600;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:4px;">This transaction\'s rank</div>'
            f'<div style="color:{COLORS["text"]};font-size:0.92rem;font-weight:600;">#{current_rank} most unusual</div>'
            f'</div>'
            f'<div style="flex:1;">'
            f'<div style="color:{COLORS["text_dim"]};font-size:0.68rem;font-weight:600;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:4px;">Anomaly level</div>'
            f'<div style="color:{multiple_color};font-size:0.92rem;font-weight:600;">{multiple_str}</div>'
            f'</div>'
            f'</div>'
        )

        steps_by_position = {s["position"]: s for s in lstm_attr["top_steps"]}
        max_err_in_top = max((s["error"] for s in lstm_attr["top_steps"]), default=1) or 1

        timeline_html = (
            f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};border-radius:8px;padding:1rem 1.2rem;margin-bottom:0.6rem;">'
            f'<div style="color:{COLORS["text_dim"]};font-size:0.68rem;font-weight:600;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:10px;">Recent activity on this card (oldest → newest)</div>'
            f'<div style="display:flex;align-items:flex-end;gap:3px;height:60px;margin-bottom:10px;">'
        )

        for pos in range(seq_len):
            step_info = steps_by_position.get(pos)
            is_curr = pos == seq_len - 1
            if step_info:
                err_normalized = step_info["error"] / max_err_in_top
                bar_height_pct = max(15, err_normalized * 100)
                bar_color = COLORS["critical"] if is_curr else COLORS["high"]
                title_text = (
                    f"step {-(seq_len - 1 - pos)} • "
                    f"${step_info.get('amount') or 0:.2f} • "
                    f"error {step_info['error']:.4f}"
                    + (" (current)" if is_curr else "")
                )
            else:
                bar_height_pct = 10
                bar_color = COLORS["border_strong"] if not is_curr else COLORS["accent"]
                title_text = f"step {-(seq_len - 1 - pos)}" + (" (current)" if is_curr else "")

            timeline_html += (
                f'<div title="{title_text}" style="flex:1;background:{bar_color};height:{bar_height_pct}%;'
                f'border-radius:2px 2px 0 0;min-width:6px;cursor:default;"></div>'
            )

        timeline_html += (
            f'</div>'
            f'<div style="display:flex;justify-content:space-between;color:{COLORS["text_dim"]};font-size:0.7rem;font-family:JetBrains Mono,monospace;">'
            f'<span>{seq_len-1} txns ago</span>'
            f'<span style="color:{COLORS["accent"]};font-weight:600;">current →</span>'
            f'</div>'
            f'<div style="display:flex;align-items:center;gap:14px;margin-top:12px;padding-top:10px;border-top:1px solid {COLORS["border"]};font-size:0.72rem;color:{COLORS["text_dim"]};">'
            f'<div style="display:flex;align-items:center;gap:5px;"><span style="display:inline-block;width:9px;height:9px;background:{COLORS["critical"]};border-radius:2px;"></span>This transaction</div>'
            f'<div style="display:flex;align-items:center;gap:5px;"><span style="display:inline-block;width:9px;height:9px;background:{COLORS["high"]};border-radius:2px;"></span>Most unusual recent transactions</div>'
            f'<div style="display:flex;align-items:center;gap:5px;"><span style="display:inline-block;width:9px;height:9px;background:{COLORS["border_strong"]};border-radius:2px;"></span>Normal-looking</div>'
            f'</div>'
            f'</div>'
        )
        _render_html(timeline_html)

        _render_html(
            f'<div style="color:{COLORS["text_dim"]};font-size:0.68rem;font-weight:600;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:8px;margin-top:14px;">Most unusual transactions in this window</div>'
        )

        for s in lstm_attr["top_steps"][:5]:
            is_current = s["is_current"]
            position = s["position"]
            steps_ago = seq_len - 1 - position
            position_label = "Current" if is_current else f"{steps_ago} txn{'s' if steps_ago > 1 else ''} ago"
            amt_v = s.get("amount")
            amt_str = f"${amt_v:.2f}" if amt_v is not None else "—"
            hour = s.get("hour")
            hour_str = (
                f"at {hour:02d}:00"
                if hour is not None and 0 <= hour <= 23 else ""
            )

            err_ratio = s["error"] / threshold if threshold > 0 else 1
            if err_ratio >= 2:
                unusualness = "Highly unusual"
                unusual_color = COLORS["critical"]
            elif err_ratio >= 1:
                unusualness = "Unusual"
                unusual_color = COLORS["high"]
            else:
                unusualness = "Mildly unusual"
                unusual_color = COLORS["medium"]

            current_pill = (
                f'<span style="background:{COLORS["critical"]};color:#0e1117;padding:2px 7px;border-radius:3px;font-size:0.62rem;font-weight:700;text-transform:uppercase;letter-spacing:0.07em;margin-left:8px;">CURRENT</span>'
                if is_current else ""
            )

            _render_html(
                f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};'
                f'{"border-left:3px solid " + COLORS["critical"] + ";" if is_current else ""}'
                f'border-radius:6px;padding:0.7rem 1rem;margin-bottom:0.4rem;'
                f'display:flex;align-items:center;justify-content:space-between;gap:14px;">'
                f'<div style="display:flex;align-items:center;gap:14px;flex:1;">'
                f'<div style="color:{COLORS["text_muted"]};font-size:0.78rem;min-width:90px;font-family:JetBrains Mono,monospace;">{position_label}</div>'
                f'<div style="color:{COLORS["text"]};font-size:0.86rem;font-weight:600;font-family:JetBrains Mono,monospace;">{amt_str}</div>'
                f'<div style="color:{COLORS["text_dim"]};font-size:0.78rem;">{hour_str}</div>'
                f'{current_pill}'
                f'</div>'
                f'<div style="color:{unusual_color};font-size:0.78rem;font-weight:600;flex-shrink:0;">{unusualness}</div>'
                f'</div>'
            )

        _render_html(
            f'<div style="color:{COLORS["text_dim"]};font-size:0.74rem;line-height:1.55;margin-top:10px;font-style:italic;">'
            f'The autoencoder learned the typical pattern of this card\'s transactions during training. '
            f'When a new transaction doesn\'t fit that pattern, it produces high reconstruction error — '
            f'the further from normal, the higher the bar.'
            f'</div>'
        )

    elif lstm_attr and lstm_attr.get("error"):
        section_header("Sequence model — recent activity check")
        _render_html(
            f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};border-left:3px solid {COLORS["medium"]};border-radius:8px;padding:0.9rem 1.1rem;margin-bottom:0.8rem;color:{COLORS["text_muted"]};font-size:0.82rem;">'
            f'LSTM timestep attribution unavailable: {lstm_attr["error"]}'
            f'</div>'
        )


# ====================================================================
# TAB 3 — Evidence
# ====================================================================
with tab_evidence:
    inv = result.get("investigator") or {}
    if inv.get("findings"):
        section_header(
            "Investigator findings",
            f"AI agent gathered evidence using {len(inv.get('tool_calls', []))} tool calls",
        )

        tool_calls = inv.get("tool_calls", [])
        if tool_calls:
            tool_pills = " ".join(
                f'<span style="background:{COLORS["surface_2"]};border:1px solid {COLORS["border"]};color:{COLORS["text_muted"]};'
                f'padding:3px 9px;border-radius:4px;font-size:0.74rem;font-family:JetBrains Mono,monospace;margin-right:5px;'
                f'display:inline-block;margin-bottom:4px;">{tc.get("tool", "?")}</span>'
                for tc in tool_calls
            )
            _render_html(
                f'<div style="margin-bottom:0.8rem;">'
                f'<div style="color:{COLORS["text_dim"]};font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:7px;">Tools called</div>'
                f'{tool_pills}'
                f'</div>'
            )

        with st.container():
            st.markdown(_escape_dollars(inv["findings"]))
    else:
        _render_html(
            f'<div style="background:{COLORS["surface"]};border:1px dashed {COLORS["border_strong"]};border-radius:8px;padding:1.4rem;text-align:center;color:{COLORS["text_muted"]};font-size:0.85rem;">'
            f'No investigator findings — the case may have been closed at triage.'
            f'</div>'
        )


# ====================================================================
# TAB 4 — Patterns
# ====================================================================
with tab_patterns:
    pat = result.get("pattern") or {}

    if pat.get("analysis"):
        section_header(
            "Pattern analysis",
            "Whether retrieved candidates actually fit this transaction",
        )
        with st.container():
            st.markdown(_escape_dollars(pat["analysis"]))
        _render_html('<div style="height:1.2rem;"></div>')

    section_header(
        "Retrieved candidates",
        "Top semantic matches from the pattern library",
    )

    if not similar:
        _render_html(
            f'<div style="background:{COLORS["surface"]};border:1px dashed {COLORS["border_strong"]};border-radius:8px;padding:1.4rem;text-align:center;">'
            f'<div style="color:{COLORS["text_muted"]};font-size:0.85rem;margin-bottom:4px;">No candidate patterns retrieved.</div>'
            f'<div style="color:{COLORS["text_dim"]};font-size:0.76rem;">This transaction\'s signature is far from anything in the library — may warrant manual review.</div>'
            f'</div>'
        )
    else:
        for case in similar:
            sim = case.get("similarity", 0)
            sim_str = f"{sim*100:.0f}%" if sim else "—"
            raw_band = case.get("match_band", "")
            if raw_band:
                band_label_map = {
                    "Strongest match": "Closest semantic match",
                    "Strong match": "Strong semantic match",
                    "Moderate match": "Moderate semantic match",
                    "Weak match": "Weak semantic match",
                }
                band = band_label_map.get(raw_band, raw_band)
            else:
                band = ""
            pattern = case.get("pattern", "unknown")
            pattern_label = pattern.replace("_", " ").title()
            case_color = PATTERN_COLORS_INVESTIGATE.get(pattern, COLORS["text_muted"])

            summary_text = case.get("snippet") or case.get("narrative", "")
            summary_text = summary_text[:280] if summary_text else ""
            reasoning_text = case.get("reasoning", "")

            _render_html(
                f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};border-left:3px solid {case_color};border-radius:8px;padding:0.95rem 1.2rem;margin-bottom:0.6rem;">'
                f'<div style="display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:8px;">'
                f'<div style="display:flex;align-items:center;gap:10px;">'
                f'<span style="color:{COLORS["text_muted"]};font-family:JetBrains Mono,monospace;font-size:0.78rem;">{case.get("id", "—")}</span>'
                f'<span style="background:{case_color};color:#0e1117;padding:2px 8px;border-radius:4px;font-size:0.66rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;">{pattern_label}</span>'
                f'</div>'
                f'<div style="display:flex;flex-direction:column;align-items:flex-end;gap:2px;flex-shrink:0;">'
                f'<span style="color:{COLORS["accent"]};font-size:0.86rem;font-weight:700;font-family:Inter,sans-serif;">{sim_str}</span>'
                f'<span style="color:{COLORS["text_dim"]};font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.06em;">{band}</span>'
                f'</div>'
                f'</div>'
                f'<div style="color:{COLORS["text"]};font-size:0.88rem;font-weight:600;margin-bottom:6px;">{case.get("title", "")}</div>'
                f'<div style="color:{COLORS["text_muted"]};font-size:0.82rem;line-height:1.55;margin-bottom:8px;">{summary_text}</div>'
                + (
                    f'<div style="background:{COLORS["surface_2"]};border-left:2px solid {COLORS["accent"]};border-radius:4px;padding:8px 12px;margin-top:8px;">'
                    f'<div style="color:{COLORS["text_dim"]};font-size:0.68rem;font-weight:600;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:4px;">Why this is fraud</div>'
                    f'<div style="color:{COLORS["text_muted"]};font-size:0.8rem;line-height:1.55;font-style:italic;">{reasoning_text}</div>'
                    f'</div>'
                    if reasoning_text else ""
                ) +
                f'</div>'
            )


# ====================================================================
# TAB 5 — Adaptive Checklist
# ====================================================================
with tab_checklist:
    pattern_analysis_text = (result.get("pattern") or {}).get("analysis", "") or ""
    pattern_analysis_lower = pattern_analysis_text.lower()

    # ----------------------------------------------------------------
    # Routing detection — robust to LLM phrasing variants
    # ----------------------------------------------------------------
    # Pattern agent emits one of three lead-sentence families:
    #
    #   Strong fit:   "[This transaction] matches the [X] pattern"
    #                 "[This transaction] strongly matches the [X] pattern"
    #                 "[This transaction] closely matches the [X] pattern"
    #                 "[This transaction] clearly matches the [X] pattern"
    #
    #   Partial fit: "[This transaction] partially matches the [X] pattern"
    #
    #   No fit:      "...none actually fits"
    #                "no pattern from the library matches"
    #                "none of the retrieved..."
    #
    # The earlier routing required the exact phrase "this transaction matches
    # the" and broke when the LLM naturally inserted "strongly" / "closely".
    # The new logic uses the substring "matches the" combined with negative
    # signals so it tolerates whatever adverb the LLM picks.
    # ----------------------------------------------------------------
    is_partial_fit = "partially matches" in pattern_analysis_lower

    has_match_phrase = (
        "matches the" in pattern_analysis_lower
        and "partially matches" not in pattern_analysis_lower
        and "none" not in pattern_analysis_lower[:200]
    )
    is_strong_fit = has_match_phrase and not is_partial_fit

    is_no_fit = (
        "none actually fits" in pattern_analysis_lower
        or "no pattern from the library matches" in pattern_analysis_lower
        or "none of the retrieved" in pattern_analysis_lower
    )

    has_real_pattern_fit = (
        bool(similar)
        and (is_strong_fit or is_partial_fit)
        and not is_no_fit
    )

    # ---- Branch A: Real pattern fit (Strong or Partial) ----
    if has_real_pattern_fit:
        # Banner showing which fit-tier the Pattern agent reached
        if is_partial_fit:
            fit_tier_label = "Partial archetype fit"
            fit_tier_color = COLORS["medium"]
            fit_tier_desc = (
                "The Pattern agent matched the archetype's story but noted some "
                "indicators differ. Below: the matched pattern's indicators, "
                "auto-verified against this transaction."
            )
        else:
            fit_tier_label = "Strong archetype fit"
            fit_tier_color = COLORS["low"]
            fit_tier_desc = (
                "The Pattern agent identified a clean match. Below: the matched "
                "pattern's indicators, auto-verified against this transaction."
            )

        section_header(
            "Pattern-grounded checklist",
            "Verifying signals from the matched fraud pattern",
        )

        _render_html(
            f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};'
            f'border-left:3px solid {fit_tier_color};border-radius:6px;'
            f'padding:0.7rem 0.95rem;margin-bottom:0.8rem;">'
            f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">'
            f'{icon("check-circle", size=14, color=fit_tier_color)}'
            f'<span style="color:{fit_tier_color};font-size:0.8rem;font-weight:600;'
            f'text-transform:uppercase;letter-spacing:0.05em;">{fit_tier_label}</span>'
            f'</div>'
            f'<div style="color:{COLORS["text_muted"]};font-size:0.78rem;line-height:1.5;">{fit_tier_desc}</div>'
            f'</div>'
        )

        coach_state_key = f"coach_result_{pick}"
        coach_btn = st.button(
            "Generate checklist",
            key="coach_btn",
            use_container_width=False,
        )

        if coach_btn:
            top_match = similar[0]
            with st.spinner(f"Building checklist from '{top_match.get('title', 'matched pattern')}'…"):
                from src.agentic.pattern_coach import build_checklist
                t0 = _time.time()
                coach_result = build_checklist(top_match, row)
                coach_duration_ms = (_time.time() - t0) * 1000

            if "error" in coach_result and not coach_result.get("checks"):
                st.error(f"Could not build checklist: {coach_result['error']}")
            else:
                st.session_state[coach_state_key] = {
                    "result": coach_result,
                    "duration_ms": coach_duration_ms,
                }

        if coach_state_key in st.session_state:
            coach_data = st.session_state[coach_state_key]
            coach_result = coach_data["result"]
            coach_duration_ms = coach_data["duration_ms"]
            checks = coach_result.get("checks", [])
            summary_dict = coach_result.get("summary", {})

            present = summary_dict.get("present", 0)
            absent = summary_dict.get("absent", 0)
            manual = summary_dict.get("manual", 0)
            unknown = summary_dict.get("unknown", 0)
            total = len(checks)

            # Five-tier indicator-overlap verdict (separate from the agent-level
            # fit-tier banner above). This summarizes how many of the matched
            # pattern's specific indicators are auto-verifiable as present here.
            verifiable = present + absent
            defining_present = (
                bool(checks)
                and checks[0].get("status") == "present"
            )

            if total == 0:
                verdict_text = "No signals to verify"
                verdict_color = COLORS["text_dim"]
            elif verifiable == 0:
                verdict_text = (
                    f"Insufficient data — {manual + unknown} of {total} signals need manual review"
                )
                verdict_color = COLORS["text_dim"]
            elif present >= 3 and present > absent:
                verdict_text = (
                    f"Strong pattern match — {present} of {total} signals present"
                )
                verdict_color = COLORS["critical"]
            elif defining_present and absent > 0:
                verdict_text = (
                    f"Defining signal present — {present} of {total} confirmed, "
                    f"{absent} diverge from pattern"
                )
                verdict_color = COLORS["medium"]
            elif present > 0:
                verdict_text = (
                    f"Mixed signals — {present} of {total} present, {absent} absent"
                )
                verdict_color = COLORS["medium"]
            elif absent >= 2:
                verdict_text = (
                    f"Pattern doesn't fit — {absent} of {total} expected signals missing"
                )
                verdict_color = COLORS["low"]
            else:
                verdict_text = (
                    f"Inconclusive — {present} present, {absent} absent, "
                    f"{manual + unknown} need review"
                )
                verdict_color = COLORS["text_dim"]

            _render_html(
                f'<div style="background:{COLORS["surface_2"]};border:1px solid {COLORS["border"]};border-left:3px solid {verdict_color};'
                f'border-radius:8px;padding:0.85rem 1.1rem;margin-bottom:1rem;display:flex;justify-content:space-between;align-items:center;gap:12px;">'
                f'<div>'
                f'<div style="color:{verdict_color};font-size:0.9rem;font-weight:600;">{verdict_text}</div>'
                f'<div style="color:{COLORS["text_dim"]};font-size:0.76rem;margin-top:3px;">'
                f'Based on: <span style="color:{COLORS["text_muted"]};">{coach_result.get("matched_pattern_title", "")}</span>'
                f'</div>'
                f'</div>'
                f'<div style="display:flex;gap:14px;align-items:center;flex-shrink:0;">'
                f'<div style="text-align:right;"><div style="color:{COLORS["low"]};font-size:1.1rem;font-weight:700;">{present}</div><div style="color:{COLORS["text_dim"]};font-size:0.7rem;text-transform:uppercase;letter-spacing:0.06em;">present</div></div>'
                f'<div style="text-align:right;"><div style="color:{COLORS["text_dim"]};font-size:1.1rem;font-weight:700;">{absent}</div><div style="color:{COLORS["text_dim"]};font-size:0.7rem;text-transform:uppercase;letter-spacing:0.06em;">absent</div></div>'
                f'<div style="text-align:right;"><div style="color:{COLORS["medium"]};font-size:1.1rem;font-weight:700;">{manual}</div><div style="color:{COLORS["text_dim"]};font-size:0.7rem;text-transform:uppercase;letter-spacing:0.06em;">manual</div></div>'
                f'</div>'
                f'</div>'
            )

            STATUS_CONFIG = {
                "present": (icon("check-circle", size=15, color=COLORS["low"]), COLORS["low"], "PRESENT"),
                "absent": (icon("x-circle", size=15, color=COLORS["text_dim"]), COLORS["text_dim"], "NOT SEEN"),
                "manual": (icon("alert-circle", size=15, color=COLORS["medium"]), COLORS["medium"], "VERIFY"),
                "unknown": (icon("info", size=15, color=COLORS["text_dim"]), COLORS["text_dim"], "NO DATA"),
            }

            for check in checks:
                status = check.get("status", "unknown")
                icon_svg, status_color, status_label = STATUS_CONFIG.get(status, STATUS_CONFIG["unknown"])
                label = check.get("label", "")
                technical = check.get("technical", "")

                actual_html = ""
                if check.get("actual_value") is not None and status in ("present", "absent"):
                    actual = check["actual_value"]
                    if isinstance(actual, float):
                        actual_str = f"{actual:.2f}"
                    else:
                        actual_str = str(actual)
                    actual_html = (
                        f'<span style="color:{COLORS["text_muted"]};font-size:0.74rem;font-family:JetBrains Mono,monospace;'
                        f'background:{COLORS["surface_2"]};padding:1px 7px;border-radius:3px;margin-left:8px;">'
                        f'value: {actual_str}</span>'
                    )

                if technical and technical != label:
                    technical_html = (
                        f'<div style="color:{COLORS["text_dim"]};font-family:JetBrains Mono,monospace;font-size:0.72rem;'
                        f'margin-top:5px;line-height:1.5;letter-spacing:-0.005em;">'
                        f'{technical}{actual_html}'
                        f'</div>'
                    )
                else:
                    technical_html = (
                        f'<div style="margin-top:5px;">{actual_html}</div>'
                        if actual_html else ""
                    )

                _render_html(
                    f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};border-radius:6px;'
                    f'padding:0.85rem 1rem;margin-bottom:0.5rem;">'
                    f'<div style="display:flex;align-items:flex-start;gap:12px;">'
                    f'<div style="flex-shrink:0;margin-top:2px;">{icon_svg}</div>'
                    f'<div style="flex:1;">'
                    f'<div style="color:{COLORS["text"]};font-size:0.86rem;font-weight:600;line-height:1.45;">{label}</div>'
                    f'{technical_html}'
                    f'</div>'
                    f'<span style="background:{status_color};color:#0e1117;padding:2px 8px;border-radius:4px;'
                    f'font-size:0.64rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;flex-shrink:0;">{status_label}</span>'
                    f'</div>'
                    f'</div>'
                )

            _render_html(
                f'<div style="color:{COLORS["text_dim"]};font-size:0.74rem;margin-top:0.6rem;">'
                f'Generated in {coach_duration_ms/1000:.1f}s. '
                f'Auto-verified checks compare against this transaction\'s features. "Verify" items need human judgment.'
                f'</div>'
            )

    # ---- Branch B: No pattern fit → SHAP-grounded with hybrid streaming ----
    else:
        attr = result.get("attribution") or {}
        xgb_attr = attr.get("xgb") or {}

        # Banner explaining why we're in Branch B (positive framing).
        # This makes the two-tier verification layer visible — semantic retrieval
        # found candidates, but the Pattern agent verified indicator fit and
        # found none clean enough to ground a checklist on.
        _render_html(
            f'<div style="background:{COLORS["info_bg"]};border:1px solid {COLORS["info"]};'
            f'border-left:3px solid {COLORS["info"]};border-radius:6px;'
            f'padding:0.7rem 0.95rem;margin-bottom:0.8rem;">'
            f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">'
            f'{icon("info", size=14, color=COLORS["info"])}'
            f'<span style="color:{COLORS["info"]};font-size:0.8rem;font-weight:600;'
            f'text-transform:uppercase;letter-spacing:0.05em;">Two-tier verification</span>'
            f'</div>'
            f'<div style="color:{COLORS["text_muted"]};font-size:0.78rem;line-height:1.5;">'
            f'Semantic retrieval found candidate patterns, but the Pattern agent\'s '
            f'indicator-fit verification found none cleanly match. Falling back to '
            f'SHAP-driven checklist grounded in the actual feature drivers below.'
            f'</div>'
            f'</div>'
        )

        section_header(
            "Model-driven verification checklist",
            "The features driving this transaction's risk score, each translated into a verification question",
        )

        shap_state_key = f"shap_coach_result_{pick}"

        if shap_state_key not in st.session_state:
            try:
                from src.agentic.shap_coach import build_shap_grounded_checklist
                t0 = _time.time()
                shap_coach_result = build_shap_grounded_checklist(xgb_attr, row.to_dict(), top_k=5)
                shap_coach_result["duration_ms"] = (_time.time() - t0) * 1000
                shap_coach_result["enriched"] = False
                shap_coach_result["enrichment_attempted"] = False
                st.session_state[shap_state_key] = shap_coach_result
            except Exception as e:
                st.error(f"Could not build SHAP-grounded checklist: {type(e).__name__}: {e}")
                st.stop()

        shap_result = st.session_state[shap_state_key]

        if not shap_result.get("enrichment_attempted"):
            with st.spinner("Tailoring verification questions to this transaction…"):
                try:
                    from src.agentic.shap_coach import enrich_checklist_with_llm
                    t0 = _time.time()
                    enriched_checks, enrich_meta = enrich_checklist_with_llm(
                        shap_result.get("checks", []), xgb_attr, row.to_dict()
                    )
                    enrich_duration_ms = (_time.time() - t0) * 1000

                    if not enrich_meta.get("error") and enrich_meta.get("enriched", 0) > 0:
                        new_summary = {"present": 0, "absent": 0, "manual": 0}
                        for c in enriched_checks:
                            s = c.get("status", "manual")
                            if s in new_summary:
                                new_summary[s] += 1
                            else:
                                new_summary["manual"] += 1
                        shap_result["checks"] = enriched_checks
                        shap_result["summary"] = new_summary
                        shap_result["enriched"] = True
                        shap_result["enrich_duration_ms"] = enrich_duration_ms
                        shap_result["enrich_meta"] = enrich_meta

                    shap_result["enrichment_attempted"] = True
                    if enrich_meta.get("error"):
                        shap_result["enrich_error"] = enrich_meta["error"]
                    st.session_state[shap_state_key] = shap_result
                except Exception as e:
                    shap_result["enrichment_attempted"] = True
                    shap_result["enrich_error"] = f"{type(e).__name__}: {e}"
                    st.session_state[shap_state_key] = shap_result

        checks = shap_result.get("checks", [])
        summary_dict = shap_result.get("summary", {})
        present = summary_dict.get("present", 0)
        absent = summary_dict.get("absent", 0)
        manual = summary_dict.get("manual", 0)
        total = len(checks)
        is_enriched = shap_result.get("enriched", False)
        enrich_error = shap_result.get("enrich_error")

        if total == 0:
            verdict_text, verdict_color = ("No signals to verify", COLORS["text_dim"])
        elif present > 0:
            verdict_text = f"{present} model-driven signal{'s' if present != 1 else ''} confirmed — review carefully"
            verdict_color = COLORS["critical"] if present >= 2 else COLORS["medium"]
        else:
            verdict_text = f"{manual} signal{'s' if manual != 1 else ''} need manual verification"
            verdict_color = COLORS["medium"]

        drivers_used = shap_result.get("drivers_used", [])[:3]
        drivers_label = ", ".join(d for d in drivers_used if d) if drivers_used else "—"

        _render_html(
            f'<div style="background:{COLORS["surface_2"]};border:1px solid {COLORS["border"]};border-left:3px solid {verdict_color};'
            f'border-radius:8px;padding:0.85rem 1.1rem;margin-bottom:1rem;display:flex;justify-content:space-between;align-items:center;gap:12px;">'
            f'<div>'
            f'<div style="color:{verdict_color};font-size:0.9rem;font-weight:600;">{verdict_text}</div>'
            f'<div style="color:{COLORS["text_dim"]};font-size:0.76rem;margin-top:3px;">'
            f'Top SHAP drivers: <span style="color:{COLORS["text_muted"]};font-family:JetBrains Mono,monospace;">{drivers_label}</span>'
            f'</div>'
            f'</div>'
            f'<div style="display:flex;gap:14px;align-items:center;flex-shrink:0;">'
            f'<div style="text-align:right;"><div style="color:{COLORS["low"]};font-size:1.1rem;font-weight:700;">{present}</div><div style="color:{COLORS["text_dim"]};font-size:0.7rem;text-transform:uppercase;letter-spacing:0.06em;">present</div></div>'
            f'<div style="text-align:right;"><div style="color:{COLORS["text_dim"]};font-size:1.1rem;font-weight:700;">{absent}</div><div style="color:{COLORS["text_dim"]};font-size:0.7rem;text-transform:uppercase;letter-spacing:0.06em;">absent</div></div>'
            f'<div style="text-align:right;"><div style="color:{COLORS["medium"]};font-size:1.1rem;font-weight:700;">{manual}</div><div style="color:{COLORS["text_dim"]};font-size:0.7rem;text-transform:uppercase;letter-spacing:0.06em;">manual</div></div>'
            f'</div>'
            f'</div>'
        )

        if enrich_error:
            _render_html(
                f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["medium"]};border-radius:6px;padding:0.7rem 0.95rem;margin-bottom:0.8rem;color:{COLORS["text_muted"]};font-size:0.78rem;">'
                f'AI tailoring couldn\'t complete (showing rule-based baseline): {enrich_error}'
                f'</div>'
            )

        STATUS_CONFIG = {
            "present": (icon("check-circle", size=15, color=COLORS["low"]), COLORS["low"], "PRESENT"),
            "absent": (icon("x-circle", size=15, color=COLORS["text_dim"]), COLORS["text_dim"], "NOT SEEN"),
            "manual": (icon("alert-circle", size=15, color=COLORS["medium"]), COLORS["medium"], "VERIFY"),
            "unknown": (icon("info", size=15, color=COLORS["text_dim"]), COLORS["text_dim"], "NO DATA"),
        }

        CATEGORY_LABELS = {
            "Model-driven check": ("Model-driven", COLORS["accent"]),
            "AI-generated check": ("AI-generated", "#a78bfa"),
            "Pending AI": ("Awaiting AI", COLORS["medium"]),
            "Baseline check": ("Baseline", COLORS["text_dim"]),
        }

        for check in checks:
            status = check.get("status", "manual")
            icon_svg, status_color, status_label = STATUS_CONFIG.get(status, STATUS_CONFIG["unknown"])
            label = check.get("label", "")
            rationale = check.get("rationale", "")
            category = check.get("category", "")
            cat_label, cat_color = CATEGORY_LABELS.get(category, (category, COLORS["text_dim"]))

            feature_pill = ""
            feature_name = check.get("feature")
            feature_value = check.get("feature_value")
            if feature_name and not str(feature_name).startswith("_baseline"):
                fv_str = f" = {feature_value}" if feature_value is not None and feature_value != "" else ""
                feature_pill = (
                    f'<span style="color:{COLORS["text_dim"]};font-size:0.7rem;font-family:JetBrains Mono,monospace;'
                    f'background:{COLORS["surface_2"]};padding:1px 6px;border-radius:3px;margin-left:6px;">'
                    f'{feature_name}{fv_str}</span>'
                )

            shap_pill = ""
            shap_v = check.get("shap_value")
            if shap_v is not None:
                shap_sign = "+" if shap_v > 0 else "−"
                shap_pill = (
                    f'<span style="color:{COLORS["text_dim"]};font-size:0.7rem;font-family:JetBrains Mono,monospace;'
                    f'background:{COLORS["surface_2"]};padding:1px 6px;border-radius:3px;margin-left:6px;">'
                    f'SHAP {shap_sign}{abs(shap_v):.2f}</span>'
                )

            rationale_html = (
                f'<div style="color:{COLORS["text_dim"]};font-size:0.76rem;line-height:1.5;">{rationale}</div>'
                if rationale else ""
            )

            _render_html(
                f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};border-radius:6px;'
                f'padding:0.85rem 1rem;margin-bottom:0.5rem;">'
                f'<div style="display:flex;align-items:flex-start;gap:12px;">'
                f'<div style="flex-shrink:0;margin-top:2px;">{icon_svg}</div>'
                f'<div style="flex:1;">'
                f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:5px;flex-wrap:wrap;">'
                f'<span style="background:{cat_color};color:#0e1117;padding:1px 7px;border-radius:3px;font-size:0.62rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;">{cat_label}</span>'
                f'{feature_pill}'
                f'{shap_pill}'
                f'</div>'
                f'<div style="color:{COLORS["text"]};font-size:0.86rem;font-weight:600;line-height:1.45;margin-bottom:3px;">{label}</div>'
                f'{rationale_html}'
                f'</div>'
                f'<span style="background:{status_color};color:#0e1117;padding:2px 8px;border-radius:4px;'
                f'font-size:0.64rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;flex-shrink:0;">{status_label}</span>'
                f'</div>'
                f'</div>'
            )

        baseline_dur = shap_result.get("duration_ms", 0)
        enrich_dur = shap_result.get("enrich_duration_ms")
        if is_enriched and enrich_dur:
            footer_text = (
                f"Baseline checklist built in {baseline_dur:.0f}ms via rule-based mapping; "
                f"enriched with AI-tailored questions in {enrich_dur/1000:.1f}s. "
                f'"Verify" items need human judgment.'
            )
        else:
            footer_text = (
                f"Built in {baseline_dur:.0f}ms from SHAP top drivers. "
                f'"Verify" items need human judgment.'
            )

        _render_html(
            f'<div style="color:{COLORS["text_dim"]};font-size:0.74rem;margin-top:0.8rem;">{footer_text}</div>'
        )
