"""Settings — admin panel for API keys, thresholds, account."""

import streamlit as st

st.set_page_config(page_title="Settings — FraudSentinel", layout="wide")

from src.dashboard import byok  # noqa: E402
from src.dashboard.components import (  # noqa: E402
    COLORS, _render_html, apply_theme, icon, page_header,
    render_login_gate, render_top_bar, section_header,
)

apply_theme()
if not render_login_gate():
    st.stop()

render_top_bar()

page_header(
    "Settings",
    "Manage your API keys, risk thresholds, and account preferences.",
)


tab_api, tab_thresholds, tab_account = st.tabs(["API & Integrations", "Risk thresholds", "Account"])


# ============= TAB 1: API KEYS =============
with tab_api:
    section_header("Anthropic API key", "Required for AI investigation")

    _render_html(
        f'<div style="color:{COLORS["text_muted"]};font-size:0.84rem;line-height:1.6;margin-bottom:1rem;">'
        f'Your key is stored only in this session and is never saved to disk. '
        f'Get an API key from <a href="https://console.anthropic.com" target="_blank" style="color:{COLORS["accent"]};">console.anthropic.com</a>.'
        f'</div>'
    )

    current_key = st.session_state.get(byok.SESSION_KEY, "")
    masked = ("•" * 20 + current_key[-4:]) if current_key else ""

    if current_key:
        _render_html(
            f'<div style="background:{COLORS["low_bg"]};border:1px solid rgba(95,201,164,0.3);border-radius:8px;padding:0.7rem 1rem;margin-bottom:0.8rem;display:flex;align-items:center;gap:10px;">'
            f'{icon("check-circle", size=14, color=COLORS["low"])}'
            f'<span style="color:{COLORS["low"]};font-size:0.85rem;font-weight:500;">Connected</span>'
            f'<span style="color:{COLORS["text_muted"]};font-family:JetBrains Mono,monospace;font-size:0.78rem;margin-left:auto;">{masked}</span>'
            f'</div>'
        )
    else:
        # Show env-fallback notice if developer has the env var set
        env_source = byok.get_api_key_source()
        if env_source == "env":
            _render_html(
                f'<div style="background:{COLORS["surface_2"]};border:1px solid {COLORS["border"]};border-radius:8px;padding:0.7rem 1rem;margin-bottom:0.8rem;display:flex;align-items:center;gap:10px;">'
                f'{icon("key", size=14, color=COLORS["text_muted"])}'
                f'<span style="color:{COLORS["text_muted"]};font-size:0.82rem;">Using ANTHROPIC_API_KEY from environment (developer mode)</span>'
                f'</div>'
            )

    new_key = st.text_input(
        "API key",
        value="",
        type="password",
        placeholder="sk-ant-...",
        label_visibility="collapsed",
        key="settings_api_key_input",
    )

    c1, c2 = st.columns([1, 4])
    with c1:
        if st.button("Save key", use_container_width=True, key="save_api_key_btn"):
            if new_key:
                st.session_state[byok.SESSION_KEY] = new_key
                st.success("API key saved for this session.")
                st.rerun()
            else:
                st.error("Please enter a key")
    with c2:
        if current_key and st.button("Remove key", key="remove_api_key_btn"):
            st.session_state.pop(byok.SESSION_KEY, None)
            st.success("API key removed.")
            st.rerun()

    section_header("Connected services")

    services = [
        {"name": "Supabase", "status": "Connected", "desc": "Authentication and user data", "ok": True},
        {"name": "Google OAuth", "status": "Connected", "desc": "Single sign-on with Google", "ok": True},
        {"name": "Anthropic API", "status": "Connected" if byok.has_api_key() else "Not connected", "desc": "Claude for AI investigation", "ok": byok.has_api_key()},
    ]

    for s in services:
        ok = s["ok"]
        c = COLORS["low"] if ok else COLORS["text_dim"]
        ic = "check-circle" if ok else "x-circle"
        _render_html(
            f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};border-radius:8px;padding:0.85rem 1.1rem;margin-bottom:0.5rem;display:flex;align-items:center;gap:14px;">'
            f'<div>{icon(ic, size=16, color=c)}</div>'
            f'<div style="flex:1;">'
            f'<div style="color:{COLORS["text"]};font-size:0.88rem;font-weight:500;">{s["name"]}</div>'
            f'<div style="color:{COLORS["text_muted"]};font-size:0.78rem;margin-top:2px;">{s["desc"]}</div>'
            f'</div>'
            f'<div style="color:{c};font-size:0.78rem;font-weight:500;">{s["status"]}</div>'
            f'</div>'
        )


# ============= TAB 2: THRESHOLDS =============
with tab_thresholds:
    section_header("Risk thresholds", "Tune when transactions are flagged for review or blocked")

    _render_html(
        f'<div style="color:{COLORS["text_muted"]};font-size:0.84rem;line-height:1.6;margin-bottom:1.2rem;">'
        f'Adjust the score cutoffs that drive your decisioning. Changes take effect immediately for this session.'
        f'</div>'
    )

    if "threshold_review" not in st.session_state:
        st.session_state["threshold_review"] = 0.6
    if "threshold_block" not in st.session_state:
        st.session_state["threshold_block"] = 0.85

    review = st.slider(
        "Send to manual review (score ≥)",
        min_value=0.1, max_value=1.0,
        value=st.session_state["threshold_review"],
        step=0.05,
        format="%.2f",
        key="t_review",
    )

    _render_html('<div style="height:0.8rem;"></div>')

    block = st.slider(
        "Auto-block (score ≥)",
        min_value=0.1, max_value=1.0,
        value=st.session_state["threshold_block"],
        step=0.05,
        format="%.2f",
        key="t_block",
    )

    if st.button("Save thresholds", key="save_thresh_btn"):
        st.session_state["threshold_review"] = review
        st.session_state["threshold_block"] = block
        st.success("Thresholds updated.")


# ============= TAB 3: ACCOUNT =============
with tab_account:
    section_header("Account")

    email = st.session_state.get("user_email", "")
    role = st.session_state.get("user_role", "Fraud Analyst")

    _render_html(
        f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};border-radius:8px;padding:1.2rem 1.4rem;margin-bottom:0.8rem;">'
        f'<div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid {COLORS["border"]};">'
        f'<span style="color:{COLORS["text_muted"]};font-size:0.82rem;">Email</span>'
        f'<span style="color:{COLORS["text"]};font-size:0.86rem;">{email}</span>'
        f'</div>'
        f'<div style="display:flex;justify-content:space-between;padding:8px 0;">'
        f'<span style="color:{COLORS["text_muted"]};font-size:0.82rem;">Role</span>'
        f'<span style="color:{COLORS["text"]};font-size:0.86rem;">{role}</span>'
        f'</div>'
        f'</div>'
    )

    section_header("Danger zone")

    from src.dashboard import auth as supabase_auth

    if st.button("Sign out of all sessions", key="signout_all_btn"):
        supabase_auth.sign_out()
        st.rerun()
        