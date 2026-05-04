"""Monitor — live transaction stream with risk-based filtering and pattern discovery."""

import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Monitor — FraudSentinel", layout="wide")

from src.dashboard.agent_log import log_discovery_run  # noqa: E402
from src.dashboard.components import (  # noqa: E402
    COLORS, _render_html, apply_theme, empty_state, icon, kpi_row,
    page_header, render_login_gate, render_top_bar, risk_band,
    section_header, style_plotly,
)
from src.dashboard.pattern_match import top_pattern_label  # noqa: E402

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


df = load_data()
if df.empty:
    st.error("No transaction data available.")
    st.stop()
df = score_all(df)


page_header(
    "Monitor",
    "Live transaction activity. New transactions appear at the top.",
)


# --- Controls ---
c1, c2, c3 = st.columns([1.2, 1.5, 2])
with c1:
    is_playing = st.session_state.get("monitor_playing", False)
    play_label = "Pause" if is_playing else "Start live feed"
    if st.button(play_label, use_container_width=True):
        st.session_state["monitor_playing"] = not is_playing
        st.rerun()

with c2:
    risk_filter = st.selectbox(
        "Risk filter",
        ["All transactions", "Medium risk and above", "High risk and above", "Critical only"],
        label_visibility="collapsed",
    )

with c3:
    speed = st.select_slider(
        "Speed", options=["Slow", "Normal", "Fast"], value="Normal",
        label_visibility="collapsed",
    )
    speed_map = {"Slow": 1.5, "Normal": 0.5, "Fast": 0.1}


filter_map = {"All transactions": 0.0, "Medium risk and above": 0.3, "High risk and above": 0.6, "Critical only": 0.85}
threshold = filter_map[risk_filter]
df_view = df[df["xgb_score"] >= threshold]

_render_html(
    f'<div style="color:{COLORS["text_muted"]};font-size:0.82rem;margin:0.8rem 0 1.2rem 0;">'
    f'Showing <span style="color:{COLORS["text"]};font-weight:500;font-family:JetBrains Mono,monospace;">{len(df_view):,}</span> '
    f'of {len(df):,} transactions</div>'
)


# --- KPI strip ---
recent_n = 100
recent = df_view.tail(recent_n) if len(df_view) > recent_n else df_view
flagged_n = int((recent["xgb_score"] >= 0.6).sum())

kpi_row([
    {"label": "Recent volume", "value": f"${recent['TransactionAmt'].sum():,.0f}", "sublabel": "last 100 transactions"},
    {"label": "Average ticket", "value": f"${recent['TransactionAmt'].mean():.2f}", "sublabel": "per transaction"},
    {"label": "Flagged", "value": f"{flagged_n}", "sublabel": f"{flagged_n / max(len(recent), 1)*100:.1f}% of recent"},
    {"label": "Highest risk", "value": f"{recent['xgb_score'].max()*100:.1f}%", "sublabel": "in current view"},
])


# ====================================================================
# Pattern Discovery Panel
# ====================================================================
section_header(
    "Find coordinated attacks",
    "Spot organized fraud rings that single-transaction scoring would miss",
)

discovery_col1, discovery_col2 = st.columns([3, 1])
with discovery_col1:
    _render_html(
        f'<div style="color:{COLORS["text_muted"]};font-size:0.84rem;line-height:1.6;">'
        f'Most fraud you see is one bad actor at a time. But sophisticated rings hit you with dozens of transactions at once — '
        f'<strong style="color:{COLORS["text"]};">same BIN range, same hour, same playbook</strong>. '
        f'Scan the last 30 high-risk transactions for signs of coordination.'
        f'</div>'
    )
with discovery_col2:
    _render_html('<div style="height:8px;"></div>')
    discovery_btn = st.button(
        "Analyze recent activity",
        use_container_width=True,
        key="discovery_btn",
    )


if discovery_btn:
    # Pull the 30 highest-risk transactions across the whole filtered view
    high_risk_df = df_view[df_view["xgb_score"] >= 0.6].nlargest(30, "xgb_score")

    if len(high_risk_df) < 3:
        _render_html(
            f'<div style="background:{COLORS["surface"]};border:1px dashed {COLORS["border_strong"]};border-radius:8px;'
            f'padding:1.4rem;text-align:center;color:{COLORS["text_muted"]};font-size:0.85rem;margin-top:1rem;">'
            f'Not enough high-risk transactions in current view ({len(high_risk_df)}/3 needed). '
            f'Try a less restrictive filter or set the risk filter to "Medium risk and above".'
            f'</div>'
        )
    else:
        with st.spinner(f"Analyzing {len(high_risk_df)} top-risk transactions for coordinated patterns…"):
            from src.agentic.pattern_discovery import discover_patterns
            t0 = time.time()
            result = discover_patterns(high_risk_df, min_score=0.6, max_txns=30)
            duration_ms = (time.time() - t0) * 1000

        if "error" in result:
            st.error(f"Pattern discovery failed: {result['error']}")
        else:
            st.session_state["discovery_result"] = result
            st.session_state["discovery_duration_ms"] = duration_ms
            st.session_state["discovery_txn_count"] = len(high_risk_df)

            try:
                log_discovery_run(
                    transactions_analyzed=result.get("transactions_analyzed", 0),
                    clusters_found=len(result.get("clusters", [])),
                    duration_ms=duration_ms,
                )
            except Exception:
                pass


if "discovery_result" in st.session_state:
    result = st.session_state["discovery_result"]
    duration_ms = st.session_state.get("discovery_duration_ms", 0)
    n_analyzed = st.session_state.get("discovery_txn_count", 0)

    _render_html('<div style="height:1rem;"></div>')

    clusters = result.get("clusters", [])
    summary = result.get("summary", "")

    if not clusters:
        _render_html(
            f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};border-left:3px solid {COLORS["low"]};'
            f'border-radius:8px;padding:1rem 1.2rem;display:flex;gap:12px;align-items:flex-start;">'
            f'<div style="flex-shrink:0;margin-top:2px;">{icon("check-circle", size=16, color=COLORS["low"])}</div>'
            f'<div>'
            f'<div style="color:{COLORS["text"]};font-size:0.9rem;font-weight:500;margin-bottom:4px;">No coordinated attacks detected</div>'
            f'<div style="color:{COLORS["text_muted"]};font-size:0.83rem;line-height:1.55;">{summary}</div>'
            f'<div style="color:{COLORS["text_dim"]};font-size:0.76rem;margin-top:8px;">Analyzed {n_analyzed} transactions in {duration_ms/1000:.1f}s.</div>'
            f'</div>'
            f'</div>'
        )
    else:
        _render_html(
            f'<div style="background:{COLORS["medium_bg"]};border:1px solid rgba(233,196,106,0.35);border-radius:8px;'
            f'padding:1rem 1.2rem;display:flex;gap:12px;align-items:flex-start;margin-bottom:1rem;">'
            f'<div style="flex-shrink:0;margin-top:2px;">{icon("alert-triangle", size=16, color=COLORS["medium"])}</div>'
            f'<div>'
            f'<div style="color:{COLORS["text"]};font-size:0.92rem;font-weight:600;margin-bottom:4px;">{len(clusters)} coordinated cluster{"s" if len(clusters) != 1 else ""} detected</div>'
            f'<div style="color:{COLORS["text_muted"]};font-size:0.85rem;line-height:1.55;">{summary}</div>'
            f'<div style="color:{COLORS["text_dim"]};font-size:0.76rem;margin-top:8px;">Analyzed {n_analyzed} transactions in {duration_ms/1000:.1f}s.</div>'
            f'</div>'
            f'</div>'
        )

        confidence_colors = {
            "high": COLORS["critical"],
            "medium": COLORS["medium"],
            "low": COLORS["info"],
        }

        for i, cluster in enumerate(clusters):
            label = cluster.get("label", "Unnamed cluster")
            pattern_type = cluster.get("pattern_type", "other")
            txn_ids = cluster.get("transaction_ids", [])
            confidence = cluster.get("confidence", "medium").lower()
            reasoning = cluster.get("reasoning", "")
            action = cluster.get("recommended_action", "")
            conf_color = confidence_colors.get(confidence, COLORS["info"])

            id_badges = " ".join(
                f'<span style="background:{COLORS["surface_2"]};border:1px solid {COLORS["border"]};color:{COLORS["text_muted"]};'
                f'padding:3px 9px;border-radius:4px;font-size:0.74rem;font-family:JetBrains Mono,monospace;'
                f'margin-right:5px;display:inline-block;margin-bottom:4px;">{tid}</span>'
                for tid in txn_ids[:12]
            )
            if len(txn_ids) > 12:
                id_badges += f'<span style="color:{COLORS["text_dim"]};font-size:0.78rem;">+{len(txn_ids)-12} more</span>'

            _render_html(
                f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};border-left:3px solid {conf_color};'
                f'border-radius:8px;padding:1.1rem 1.3rem;margin-bottom:0.8rem;">'
                f'<div style="display:flex;justify-content:space-between;align-items:flex-start;gap:14px;margin-bottom:10px;">'
                f'<div style="color:{COLORS["text"]};font-size:0.95rem;font-weight:600;letter-spacing:-0.01em;">{label}</div>'
                f'<div style="display:flex;gap:10px;align-items:center;flex-shrink:0;">'
                f'<span style="background:{conf_color};color:#0e1117;padding:2px 8px;border-radius:4px;font-size:0.66rem;font-weight:700;'
                f'text-transform:uppercase;letter-spacing:0.06em;">{confidence} conf</span>'
                f'<span style="color:{COLORS["text_muted"]};font-size:0.74rem;font-family:JetBrains Mono,monospace;">{pattern_type}</span>'
                f'</div>'
                f'</div>'
                f'<div style="color:{COLORS["text_muted"]};font-size:0.86rem;line-height:1.6;margin-bottom:12px;">{reasoning}</div>'
                f'<div style="margin-bottom:10px;">'
                f'<div style="color:{COLORS["text_dim"]};font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:7px;">Transactions in this cluster ({len(txn_ids)})</div>'
                f'{id_badges}'
                f'</div>'
                f'<div style="background:{COLORS["accent_bg"]};border:1px solid rgba(94,234,212,0.25);border-radius:6px;padding:0.7rem 0.95rem;display:flex;gap:10px;align-items:flex-start;">'
                f'<div style="flex-shrink:0;margin-top:1px;">{icon("zap", size=12, color=COLORS["accent"])}</div>'
                f'<div style="color:{COLORS["text"]};font-size:0.83rem;line-height:1.5;"><strong style="color:{COLORS["accent"]};">Recommended:</strong> {action}</div>'
                f'</div>'
                f'</div>'
            )

    _render_html('<div style="height:0.6rem;"></div>')
    if st.button("Clear analysis", key="clear_discovery"):
        for k in ("discovery_result", "discovery_duration_ms", "discovery_txn_count"):
            st.session_state.pop(k, None)
        st.rerun()


# --- Activity chart ---
section_header("Activity", "Each dot is a transaction. Larger dots are higher amounts.")

chart_df = df_view.tail(500).copy().reset_index(drop=True)
chart_df["row_idx"] = range(len(chart_df))

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=chart_df["row_idx"],
        y=chart_df["xgb_score"],
        mode="markers",
        marker=dict(
            size=chart_df["TransactionAmt"].clip(0, 2000) / 80 + 4,
            color=chart_df["xgb_score"],
            colorscale=[
                [0.0, COLORS["low"]],
                [0.3, COLORS["medium"]],
                [0.6, COLORS["high"]],
                [1.0, COLORS["critical"]],
            ],
            showscale=False,
            line=dict(width=0),
            opacity=0.85,
        ),
        hovertemplate="<b>Txn %{customdata[0]}</b><br>$%{customdata[1]:.2f}<br>Risk: %{y:.1%}<extra></extra>",
        customdata=list(zip(chart_df.get("TransactionID", chart_df["row_idx"]), chart_df["TransactionAmt"])),
    )
)
fig.add_hline(y=0.6, line_dash="dot", line_color=COLORS["high"], opacity=0.4)
fig.add_hline(y=0.85, line_dash="dot", line_color=COLORS["critical"], opacity=0.4)
fig.update_layout(
    xaxis=dict(title=dict(text="Transactions over time →", font=dict(size=10))),
    yaxis=dict(title=dict(text="Risk score", font=dict(size=10)), tickformat=".0%", range=[0, 1.05]),
)
style_plotly(fig, height=360)
st.plotly_chart(fig, use_container_width=True)


# --- Recent transactions with pattern badges ---
section_header("Recent transactions", "Click any transaction to investigate")

display_n = st.slider("Rows to show", 10, 200, 30, step=10)
recent_df = df_view.tail(display_n).iloc[::-1]


def render_transactions_with_patterns(df_view, max_rows=30):
    if df_view is None or len(df_view) == 0:
        empty_state("No transactions match current filters")
        return

    PATTERN_COLORS_LOCAL = {
        "Card Testing": COLORS["critical"],
        "Geo Anomaly": COLORS["info"],
        "Account Takeover": "#a78bfa",
        "Velocity Attack": COLORS["high"],
        "Synthetic Identity": COLORS["accent"],
        "Bin Attack": COLORS["medium"],
        "Friendly Fraud": COLORS["low"],
        "Temporal Anomaly": "#f0a4d2",
        "Email Risk": "#84d4b8",
        "Subscription Probe": "#9aafd4",
    }

    df_view = df_view.head(max_rows).copy()
    rows = []

    for _, row in df_view.iterrows():
        score = float(row.get("xgb_score", 0))
        label, color = risk_band(score)
        amt = float(row.get("TransactionAmt", 0))
        product = str(row.get("ProductCD", "?"))
        txn_id = str(row.get("TransactionID", ""))
        card_id = str(row.get("card1", ""))

        pattern_html = ""
        if score >= 0.6:
            try:
                pattern = top_pattern_label(row)
                if pattern:
                    p_color = PATTERN_COLORS_LOCAL.get(pattern, COLORS["text_muted"])
                    pattern_html = (
                        f'<span style="background:{p_color};color:#0e1117;padding:2px 7px;border-radius:4px;'
                        f'font-size:0.66rem;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;">{pattern}</span>'
                    )
            except Exception:
                pass
        if not pattern_html:
            pattern_html = f'<span style="color:{COLORS["text_dim"]};font-size:0.78rem;">—</span>'

        rows.append(
            f'<tr style="border-bottom:1px solid {COLORS["border"]};">'
            f'<td style="padding:9px 12px;">'
            f'<span style="display:inline-block;width:5px;height:5px;border-radius:50%;background:{color};margin-right:8px;vertical-align:1px;"></span>'
            f'<span style="color:{color};font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;">{label}</span>'
            f'<span style="color:{COLORS["text_muted"]};font-family:JetBrains Mono,monospace;font-size:0.76rem;margin-left:8px;">{score*100:.1f}%</span>'
            f'</td>'
            f'<td style="padding:9px 12px;color:{COLORS["text_muted"]};font-family:JetBrains Mono,monospace;font-size:0.8rem;">{txn_id}</td>'
            f'<td style="padding:9px 12px;color:{COLORS["text"]};font-feature-settings:tnum;text-align:right;">${amt:,.2f}</td>'
            f'<td style="padding:9px 12px;color:{COLORS["text_muted"]};font-family:JetBrains Mono,monospace;font-size:0.8rem;">{card_id}</td>'
            f'<td style="padding:9px 12px;color:{COLORS["text_muted"]};font-size:0.84rem;">{product}</td>'
            f'<td style="padding:9px 12px;">{pattern_html}</td>'
            f'</tr>'
        )

    head_cell = f'padding:10px 12px;color:{COLORS["text_muted"]};font-size:0.68rem;font-weight:600;text-transform:uppercase;letter-spacing:0.06em;'
    table_html = (
        f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};border-radius:8px;overflow:hidden;">'
        f'<table style="width:100%;border-collapse:collapse;">'
        f'<thead style="background:{COLORS["surface_2"]};">'
        f'<tr>'
        f'<th style="{head_cell}text-align:left;">Risk</th>'
        f'<th style="{head_cell}text-align:left;">Transaction</th>'
        f'<th style="{head_cell}text-align:right;">Amount</th>'
        f'<th style="{head_cell}text-align:left;">Card</th>'
        f'<th style="{head_cell}text-align:left;">Channel</th>'
        f'<th style="{head_cell}text-align:left;">Looks like</th>'
        f'</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody>'
        f'</table></div>'
    )
    _render_html(table_html)


render_transactions_with_patterns(recent_df, max_rows=display_n)


# Quick-investigate buttons
_render_html('<div style="height:14px;"></div>')
_render_html(f'<div style="color:{COLORS["text_muted"]};font-size:0.82rem;margin-bottom:8px;">Quick actions on top-risk transactions:</div>')

top_inv = recent_df.nlargest(min(6, len(recent_df)), "xgb_score")
btn_cols = st.columns(min(6, len(top_inv)))
for i, (_, row) in enumerate(top_inv.iterrows()):
    with btn_cols[i]:
        tid = str(row.get("TransactionID", f"T{i}"))
        if st.button(f"Investigate {tid}", key=f"inv_btn_{i}_{tid}", use_container_width=True):
            st.session_state["selected_transaction_id"] = tid
            st.switch_page("pages/2_Investigate.py")


if st.session_state.get("monitor_playing"):
    time.sleep(speed_map[speed])
    st.session_state["monitor_cursor"] = (st.session_state.get("monitor_cursor", 0) + 1) % len(df)
    st.rerun()
    