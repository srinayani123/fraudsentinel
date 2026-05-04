"""FraudSentinel — main entrypoint."""

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="FraudSentinel",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.dashboard.components import (  # noqa: E402
    COLORS, _render_html, apply_theme, icon, kpi_row,
    page_header, render_login_gate, render_top_bar, section_header,
)
from src.utils.config import SAMPLES_DIR  # noqa: E402

apply_theme()

if not render_login_gate():
    st.stop()


@st.cache_data
def load_sample_transactions() -> pd.DataFrame:
    path = SAMPLES_DIR / "demo_transactions.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def score_all_transactions(_df: pd.DataFrame) -> pd.DataFrame:
    if _df.empty:
        return _df
    try:
        from src.ml_models.inference import score_dataframe
        df = _df.copy()
        df["xgb_score"] = score_dataframe(df)
        return df
    except Exception:
        df = _df.copy()
        df["xgb_score"] = 0.0
        return df


with st.sidebar:
    _render_html(
        f'<div style="display:flex;align-items:center;gap:10px;padding:8px 0 14px 0;border-bottom:1px solid {COLORS["border"]};margin-bottom:14px;">'
        f'<div style="background:{COLORS["accent"]};width:30px;height:30px;border-radius:7px;display:flex;align-items:center;justify-content:center;">'
        f'{icon("shield", size=16, color="#0e1117")}</div>'
        f'<div>'
        f'<div style="color:{COLORS["text"]};font-weight:700;font-size:0.95rem;letter-spacing:-0.01em;line-height:1.1;">FraudSentinel</div>'
        f'<div style="color:{COLORS["text_dim"]};font-size:0.66rem;text-transform:uppercase;letter-spacing:0.08em;margin-top:2px;">Risk Operations</div>'
        f'</div></div>'
    )

render_top_bar()

page_header(
    "Today's overview",
    "Activity across your transaction stream.",
)

df = load_sample_transactions()
df = score_all_transactions(df) if not df.empty else df

if df.empty:
    _render_html(
        f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};border-left:3px solid {COLORS["medium"]};border-radius:6px;padding:1rem 1.2rem;">'
        f'<div style="color:{COLORS["text"]};font-weight:600;margin-bottom:4px;font-size:0.92rem;">No transaction data available</div>'
        f'<div style="color:{COLORS["text_muted"]};font-size:0.84rem;line-height:1.5;">Connect your data source to see activity.</div>'
        f'</div>'
    )
    st.stop()


total = len(df)
high_risk = int((df["xgb_score"] >= 0.6).sum())
critical = int((df["xgb_score"] >= 0.85).sum())
total_amount = float(df["TransactionAmt"].sum())
flagged_amount = float(df[df["xgb_score"] >= 0.6]["TransactionAmt"].sum())
approved = total - high_risk
approval_rate = approved / total if total else 0

kpi_row([
    {"label": "Transactions", "value": f"{total:,}", "sublabel": "today"},
    {"label": "Approval rate", "value": f"{approval_rate*100:.1f}%", "sublabel": f"{approved:,} approved"},
    {"label": "Pending review", "value": f"{high_risk:,}", "sublabel": "high-risk"},
    {"label": "Total volume", "value": f"${total_amount/1000:,.0f}K", "sublabel": "processed today"},
    {"label": "At-risk volume", "value": f"${flagged_amount/1000:,.0f}K", "sublabel": f"{flagged_amount/total_amount*100:.1f}% flagged"},
])


section_header("Workspaces")


def nav_card(title: str, description: str, icon_name: str, page_path: str, key: str):
    _render_html(
        f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};border-radius:8px;padding:1.1rem;height:100%;display:flex;flex-direction:column;gap:8px;min-height:140px;">'
        f'<div style="display:flex;align-items:center;gap:10px;">'
        f'<div style="background:{COLORS["accent_bg"]};border-radius:6px;padding:6px;display:flex;">'
        f'{icon(icon_name, size=14, color=COLORS["accent"])}</div>'
        f'<div style="color:{COLORS["text"]};font-weight:600;font-size:0.92rem;">{title}</div>'
        f'</div>'
        f'<div style="color:{COLORS["text_muted"]};font-size:0.82rem;line-height:1.5;flex-grow:1;">{description}</div>'
        f'</div>'
    )
    if st.button(f"Open {title}", key=key, use_container_width=True):
        st.switch_page(page_path)


modules_row1 = [
    {"title": "Monitor", "description": "Live feed of transactions as they're scored. Filter by risk level and drill in.", "icon": "activity", "path": "pages/1_Monitor.py", "key": "nav_monitor"},
    {"title": "Investigate", "description": "Review flagged transactions. AI agents help you reach a decision faster.", "icon": "search", "path": "pages/2_Investigate.py", "key": "nav_investigate"},
    {"title": "Assistant", "description": "Ask questions about transactions, cards, or fraud patterns in plain English.", "icon": "message-circle", "path": "pages/6_Assistant.py", "key": "nav_chat"},
]
modules_row2 = [
    {"title": "Insights", "description": "Performance trends, latency, and what's driving recent decisions.", "icon": "bar-chart", "path": "pages/3_Insights.py", "key": "nav_insights"},
    {"title": "Test", "description": "Score sample transactions to validate rules before promoting them to production.", "icon": "flask", "path": "pages/4_Test.py", "key": "nav_test"},
    {"title": "Pattern library", "description": "Curated catalog of historical fraud patterns. Used by the AI to find similar cases.", "icon": "book", "path": "pages/5_Pattern_Library.py", "key": "nav_kb"},
]

cols = st.columns(3)
for col, mod in zip(cols, modules_row1):
    with col:
        nav_card(mod["title"], mod["description"], mod["icon"], mod["path"], mod["key"])

_render_html('<div style="height:14px;"></div>')

cols = st.columns(3)
for col, mod in zip(cols, modules_row2):
    with col:
        nav_card(mod["title"], mod["description"], mod["icon"], mod["path"], mod["key"])
        