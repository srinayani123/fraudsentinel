"""
FraudSentinel design system. Linear/Raycast-inspired warm dark UI.
"""

from __future__ import annotations

from typing import Optional

import streamlit as st


COLORS = {
    "bg": "#0e1117", "surface": "#161b22", "surface_2": "#1c2230", "surface_3": "#242b3a",
    "border": "#262e3d", "border_strong": "#363f52",
    "text": "#eaeef3", "text_muted": "#9ba3b3", "text_dim": "#6b7385", "text_subtle": "#4b5260",
    "critical": "#ef6f6c", "critical_bg": "rgba(239, 111, 108, 0.10)",
    "high": "#f5a25d", "high_bg": "rgba(245, 162, 93, 0.10)",
    "medium": "#e9c46a", "medium_bg": "rgba(233, 196, 106, 0.10)",
    "low": "#5fc9a4", "low_bg": "rgba(95, 201, 164, 0.10)",
    "info": "#7eb6e8", "info_bg": "rgba(126, 182, 232, 0.10)",
    "trend_up_good": "#5fc9a4", "trend_down_bad": "#ef6f6c", "trend_neutral": "#6b7385",
    "accent": "#5eead4", "accent_hover": "#4ad6c0",
    "accent_bg": "rgba(94, 234, 212, 0.10)", "accent_dim": "rgba(94, 234, 212, 0.5)",
}

CHART_BG = "#161b22"
CHART_GRID = "#262e3d"
CHART_TEXT = "#9ba3b3"
CHART_PALETTE = ["#5eead4", "#ef6f6c", "#f5a25d", "#a78bfa", "#5fc9a4", "#7eb6e8"]


def _render_html(html: str):
    compact = " ".join(line.strip() for line in html.strip().split("\n") if line.strip())
    if hasattr(st, "html"):
        st.html(compact)
    else:
        st.markdown(compact, unsafe_allow_html=True)


def apply_theme():
    if st.session_state.get("_theme_applied"):
        return
    st.session_state["_theme_applied"] = True

    css = (
        "<style>"
        "@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');"
        f".stApp {{ background: {COLORS['bg']}; color: {COLORS['text']}; font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }}"
        f"section[data-testid=\"stSidebar\"] {{ background: {COLORS['surface']}; border-right: 1px solid {COLORS['border']}; }}"
        "section[data-testid=\"stSidebar\"] > div { padding-top: 0.5rem; }"
        f"h1, h2, h3, h4, h5, h6 {{ font-family: 'Inter', sans-serif; color: {COLORS['text']}; font-weight: 600; letter-spacing: -0.015em; }}"
        "h1 { font-size: 1.4rem !important; margin-bottom: 0.25rem !important; line-height: 1.2; }"
        "h2 { font-size: 1.15rem !important; }"
        "h3 { font-size: 1rem !important; }"
        "p, span, div, label { font-family: 'Inter', sans-serif; }"
        "code, pre, .stCode { font-family: 'JetBrains Mono', ui-monospace, monospace !important; }"
        f"section[data-testid=\"stSidebar\"] a[data-testid=\"stSidebarNavLink\"] {{ border-radius: 6px; margin: 1px 0; padding: 7px 12px; font-size: 0.86rem; color: {COLORS['text_muted']}; }}"
        f"section[data-testid=\"stSidebar\"] a[data-testid=\"stSidebarNavLink\"]:hover {{ background: {COLORS['surface_2']}; color: {COLORS['text']}; }}"
        f"section[data-testid=\"stSidebar\"] a[data-testid=\"stSidebarNavLink\"][aria-current=\"page\"] {{ background: {COLORS['surface_2']}; color: {COLORS['text']}; font-weight: 500; }}"
        f".stTextInput input, .stNumberInput input, .stSelectbox > div > div, .stTextArea textarea {{ background: {COLORS['surface_2']} !important; border: 1px solid {COLORS['border']} !important; border-radius: 6px !important; color: {COLORS['text']} !important; font-family: 'Inter', sans-serif !important; font-size: 0.86rem !important; padding: 8px 12px !important; }}"
        f".stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus {{ border-color: {COLORS['accent']} !important; box-shadow: 0 0 0 2px rgba(94,234,212,0.18) !important; }}"
        f".stButton > button {{ background: {COLORS['accent']}; color: #0e1117; border: none; border-radius: 6px; padding: 0.5rem 1rem; font-size: 0.86rem; font-weight: 600; font-family: 'Inter', sans-serif; min-height: 38px; box-shadow: 0 1px 2px rgba(0,0,0,0.2); }}"
        f".stButton > button:hover {{ background: {COLORS['accent_hover']}; color: #0e1117; }}"
        ".stButton > button:active { transform: translateY(1px); }"
        f".stButton > button:disabled {{ background: {COLORS['surface_2']} !important; color: {COLORS['text_dim']} !important; }}"
        f"div[data-testid=\"stButton\"] button[kind=\"secondary\"] {{ background: white !important; color: #1f2937 !important; border: 1px solid #d1d5db !important; font-weight: 500 !important; }}"
        f"div[data-testid=\"stButton\"] button[kind=\"secondary\"]:hover {{ background: #f9fafb !important; color: #111827 !important; border-color: #9ca3af !important; }}"
        f".stMetric {{ background: {COLORS['surface']}; border: 1px solid {COLORS['border']}; border-radius: 8px; padding: 1rem 1.2rem; }}"
        f".stMetric label {{ color: {COLORS['text_muted']} !important; font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600; }}"
        f".stMetric [data-testid=\"stMetricValue\"] {{ color: {COLORS['text']} !important; font-size: 1.55rem !important; font-weight: 700 !important; font-family: 'Inter', sans-serif !important; }}"
        f".stDataFrame {{ border: 1px solid {COLORS['border']}; border-radius: 8px; }}"
        f"button[role=\"tab\"] {{ color: {COLORS['text_muted']} !important; font-family: 'Inter', sans-serif !important; font-weight: 500 !important; font-size: 0.86rem !important; }}"
        f"button[role=\"tab\"][aria-selected=\"true\"] {{ color: {COLORS['text']} !important; border-bottom-color: {COLORS['accent']} !important; }}"
        "[data-testid=\"stToolbar\"] { display: none !important; }"
        "footer { display: none !important; }"
        "#MainMenu { display: none !important; }"
        "[data-testid=\"stStatusWidget\"] { display: none !important; }"
        ".block-container { padding-top: 1.4rem !important; padding-bottom: 2rem !important; padding-left: 2rem !important; padding-right: 2rem !important; max-width: 1380px !important; }"
        f"hr {{ border: none !important; border-top: 1px solid {COLORS['border']} !important; }}"
        f".stSlider [data-baseweb=\"slider\"] [role=\"slider\"] {{ background: {COLORS['accent']} !important; }}"
        f".stCaption {{ color: {COLORS['text_dim']} !important; font-size: 0.78rem !important; }}"
        f"div[data-testid=\"stAlert\"] {{ border-radius: 8px !important; padding: 0.7rem 0.95rem !important; font-size: 0.85rem !important; }}"
        f"div[data-baseweb=\"notification\"][kind=\"error\"] {{ background: {COLORS['critical_bg']} !important; border: 1px solid rgba(239,111,108,0.3) !important; color: #ffb3b1 !important; }}"
        f"div[data-baseweb=\"notification\"][kind=\"info\"] {{ background: {COLORS['info_bg']} !important; border: 1px solid rgba(126,182,232,0.3) !important; color: #b8d6f3 !important; }}"
        f"div[data-baseweb=\"notification\"][kind=\"success\"] {{ background: {COLORS['low_bg']} !important; border: 1px solid rgba(95,201,164,0.3) !important; color: #9eddc7 !important; }}"
        f"div[data-baseweb=\"notification\"][kind=\"warning\"] {{ background: {COLORS['medium_bg']} !important; border: 1px solid rgba(233,196,106,0.3) !important; color: #f3dba0 !important; }}"
        "div[data-testid=\"stButton\"] button[kind=\"secondary\"][data-google=\"1\"] { padding-left: 38px !important; background-repeat: no-repeat !important; background-position: 14px center !important; background-size: 18px 18px !important; "
        "background-image: url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 48 48'><path fill='%23FFC107' d='M43.611 20.083H42V20H24v8h11.303c-1.649 4.657-6.08 8-11.303 8-6.627 0-12-5.373-12-12s5.373-12 12-12c3.059 0 5.842 1.154 7.961 3.039l5.657-5.657C34.046 6.053 29.268 4 24 4 12.955 4 4 12.955 4 24s8.955 20 20 20 20-8.955 20-20c0-1.341-.138-2.65-.389-3.917z'/><path fill='%23FF3D00' d='M6.306 14.691l6.571 4.819C14.655 15.108 18.961 12 24 12c3.059 0 5.842 1.154 7.961 3.039l5.657-5.657C34.046 6.053 29.268 4 24 4 16.318 4 9.656 8.337 6.306 14.691z'/><path fill='%234CAF50' d='M24 44c5.166 0 9.86-1.977 13.409-5.192l-6.19-5.238A11.91 11.91 0 0124 36c-5.202 0-9.619-3.317-11.283-7.946l-6.522 5.025C9.505 39.556 16.227 44 24 44z'/><path fill='%231976D2' d='M43.611 20.083H42V20H24v8h11.303a12.04 12.04 0 01-4.087 5.571l.003-.002 6.19 5.238C36.971 39.205 44 34 44 24c0-1.341-.138-2.65-.389-3.917z'/></svg>\") !important; "
        "background-color: white !important; }"
        "@media (max-width: 1024px) {"
        "  .block-container { padding-left: 1.2rem !important; padding-right: 1.2rem !important; padding-top: 1rem !important; }"
        "  h1 { font-size: 1.25rem !important; }"
        "  .stMetric [data-testid=\"stMetricValue\"] { font-size: 1.3rem !important; }"
        "}"
        "@media (max-width: 768px) {"
        "  .block-container { padding-left: 0.8rem !important; padding-right: 0.8rem !important; padding-top: 0.8rem !important; }"
        "  h1 { font-size: 1.15rem !important; }"
        "  .stMetric { padding: 0.7rem 0.85rem !important; }"
        "  div[data-testid=\"column\"] { width: 100% !important; flex: 1 1 100% !important; min-width: 100% !important; }"
        "  div[data-testid=\"stHorizontalBlock\"] { flex-wrap: wrap !important; gap: 0.5rem !important; }"
        "}"
        "div[data-testid=\"stHtmlBlock\"] { overflow-x: auto; }"
        "</style>"
    )
    st.markdown(css, unsafe_allow_html=True)


def inject_global_styles():
    apply_theme()


def risk_band(score: float):
    if score >= 0.85: return ("CRITICAL", COLORS["critical"])
    if score >= 0.6:  return ("HIGH", COLORS["high"])
    if score >= 0.3:  return ("MEDIUM", COLORS["medium"])
    return ("LOW", COLORS["low"])


def risk_band_full(score: float):
    if score >= 0.85: return ("CRITICAL", "critical", COLORS["critical"])
    if score >= 0.6:  return ("HIGH", "high", COLORS["high"])
    if score >= 0.3:  return ("MEDIUM", "medium", COLORS["medium"])
    return ("LOW", "low", COLORS["low"])


LUCIDE_PATHS = {
    "alert-triangle": '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    "alert-circle": '<circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>',
    "shield": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>',
    "shield-check": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="M9 12l2 2 4-4"/>',
    "activity": '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
    "search": '<circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/>',
    "bar-chart": '<path d="M12 20V10"/><path d="M18 20V4"/><path d="M6 20v-6"/>',
    "trending-up": '<polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/>',
    "trending-down": '<polyline points="22 17 13.5 8.5 8.5 13.5 2 7"/><polyline points="16 17 22 17 22 11"/>',
    "minus": '<line x1="5" y1="12" x2="19" y2="12"/>',
    "message-square": '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>',
    "message-circle": '<path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"/>',
    "book": '<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>',
    "credit-card": '<rect x="2" y="5" width="20" height="14" rx="2"/><line x1="2" y1="10" x2="22" y2="10"/>',
    "user": '<path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
    "users": '<path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "check-circle": '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>',
    "x-circle": '<circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/>',
    "info": '<circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>',
    "settings": '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>',
    "zap": '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
    "filter": '<polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/>',
    "refresh-cw": '<polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>',
    "send": '<line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/>',
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>',
    "file-text": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/>',
    "cpu": '<rect x="4" y="4" width="16" height="16" rx="2" ry="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/>',
    "eye": '<path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/>',
    "chevron-right": '<polyline points="9 18 15 12 9 6"/>',
    "chevron-down": '<polyline points="6 9 12 15 18 9"/>',
    "arrow-right": '<line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/>',
    "tool": '<path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/>',
    "key": '<path d="M21 2l-2 2m-7.61 7.61a5.5 5.5 0 1 1-7.778 7.778 5.5 5.5 0 0 1 7.777-7.777zm0 0L15.5 7.5m0 0l3 3L22 7l-3-3m-3.5 3.5L19 4"/>',
    "log-out": '<path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><polyline points="16 17 21 12 16 7"/><line x1="21" y1="12" x2="9" y2="12"/>',
    "lock": '<rect x="3" y="11" width="18" height="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/>',
    "layers": '<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>',
    "list": '<line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/>',
    "calendar": '<rect x="3" y="4" width="18" height="18" rx="2" ry="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/>',
    "dollar-sign": '<line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>',
    "play": '<polygon points="5 3 19 12 5 21 5 3"/>',
    "pause": '<rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/>',
    "more-horizontal": '<circle cx="12" cy="12" r="1"/><circle cx="19" cy="12" r="1"/><circle cx="5" cy="12" r="1"/>',
    "external-link": '<path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/>',
    "sparkles": '<path d="M12 3l1.5 4.5L18 9l-4.5 1.5L12 15l-1.5-4.5L6 9l4.5-1.5z"/><path d="M19 15l1 3 3 1-3 1-1 3-1-3-3-1 3-1z"/>',
    "flask": '<path d="M9 2h6"/><path d="M10 2v5L4 20a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1L14 7V2"/>',
}


def icon(name: str, size: int = 16, color: Optional[str] = None) -> str:
    if color is None:
        color = COLORS["text_muted"]
    path = LUCIDE_PATHS.get(name)
    if not path:
        return f'<span style="display:inline-block;width:{size}px;height:{size}px;"></span>'
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round" '
        f'style="display:inline-block;vertical-align:-3px;">{path}</svg>'
    )


def render_icon(name: str, size: int = 16, color: Optional[str] = None):
    _render_html(icon(name, size, color))


def page_header(title: str, subtitle: Optional[str] = None, badge: Optional[str] = None):
    sub = f'<p style="color:{COLORS["text_muted"]};font-size:0.86rem;margin:0;line-height:1.4;">{subtitle}</p>' if subtitle else ""
    html = (
        f'<div style="margin-bottom:1.4rem;">'
        f'<h1 style="margin:0 0 0.3rem 0;color:{COLORS["text"]};font-size:1.4rem;font-weight:600;letter-spacing:-0.015em;">{title}</h1>'
        f'{sub}'
        f'<div style="border-bottom:1px solid {COLORS["border"]};margin-top:0.9rem;"></div>'
        f'</div>'
    )
    _render_html(html)


def section_header(title: str, subtitle: Optional[str] = None, icon_name: Optional[str] = None):
    sub = f'<span style="color:{COLORS["text_muted"]};font-size:0.8rem;margin-left:10px;font-weight:400;">{subtitle}</span>' if subtitle else ""
    html = (
        f'<div style="margin:1.3rem 0 0.7rem 0;">'
        f'<span style="color:{COLORS["text"]};font-size:0.92rem;font-weight:600;letter-spacing:-0.01em;">{title}</span>'
        f'{sub}'
        f'</div>'
    )
    _render_html(html)


def status_pill(label: str, color_key: str = "info") -> str:
    color = COLORS.get(color_key, COLORS["info"])
    bg = COLORS.get(f"{color_key}_bg", COLORS["info_bg"])
    return (
        f'<span style="display:inline-block;padding:2px 8px;background:{bg};color:{color};'
        f'border-radius:4px;font-size:0.7rem;font-weight:600;text-transform:uppercase;'
        f'letter-spacing:0.04em;font-family:Inter,sans-serif;">{label}</span>'
    )


def render_status_pill(label: str, color_key: str = "info"):
    _render_html(status_pill(label, color_key))


def risk_pill(score: float) -> str:
    label, color_key, color = risk_band_full(score)
    bg = COLORS.get(f"{color_key}_bg", COLORS["info_bg"])
    return (
        f'<span style="display:inline-flex;align-items:center;gap:6px;padding:3px 9px;'
        f'background:{bg};color:{color};border-radius:4px;font-size:0.7rem;font-weight:700;'
        f'text-transform:uppercase;letter-spacing:0.05em;font-family:Inter,sans-serif;">'
        f'<span style="display:inline-block;width:5px;height:5px;border-radius:50%;background:{color};"></span>'
        f'{label}<span style="color:{COLORS["text_muted"]};font-family:JetBrains Mono,monospace;font-weight:500;margin-left:4px;font-size:0.72rem;">{score*100:.1f}%</span></span>'
    )


def risk_indicator(score: float):
    _render_html(risk_pill(score))


def metric_card(label: str, value: str, delta: Optional[str] = None, delta_positive: Optional[bool] = None, sublabel: Optional[str] = None):
    sub_html = f'<div style="color:{COLORS["text_dim"]};font-size:0.74rem;margin-top:5px;line-height:1.3;">{sublabel}</div>' if sublabel else ""
    delta_html = ""
    if delta:
        if delta_positive is True:
            d_color = COLORS["trend_up_good"]; arrow = icon("trending-up", size=12, color=d_color)
        elif delta_positive is False:
            d_color = COLORS["trend_down_bad"]; arrow = icon("trending-down", size=12, color=d_color)
        else:
            d_color = COLORS["trend_neutral"]; arrow = icon("minus", size=12, color=d_color)
        delta_html = (
            f'<div style="display:inline-flex;align-items:center;gap:4px;color:{d_color};'
            f'font-size:0.74rem;font-weight:600;margin-top:5px;">{arrow}<span>{delta}</span></div>'
        )
    html = (
        f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};'
        f'border-radius:8px;padding:1rem 1.15rem;min-height:96px;">'
        f'<div style="color:{COLORS["text_muted"]};font-size:0.68rem;font-weight:600;'
        f'text-transform:uppercase;letter-spacing:0.07em;margin-bottom:8px;">{label}</div>'
        f'<div style="color:{COLORS["text"]};font-size:1.5rem;font-weight:700;font-family:Inter,sans-serif;line-height:1.1;letter-spacing:-0.02em;">{value}</div>'
        f'{sub_html}{delta_html}</div>'
    )
    _render_html(html)


def metric_card_with_sparkline(label: str, value: str, sparkline_values: list,
                                delta: Optional[str] = None, delta_positive: Optional[bool] = None,
                                sparkline_color: Optional[str] = None):
    """KPI card with embedded inline SVG sparkline. sparkline_values = list of numbers."""
    if sparkline_color is None:
        sparkline_color = COLORS["accent"]

    if sparkline_values and len(sparkline_values) > 1:
        vmin, vmax = min(sparkline_values), max(sparkline_values)
        rng = vmax - vmin if vmax != vmin else 1
        w, h = 100, 30
        pad = 2
        n = len(sparkline_values)
        points = []
        for i, v in enumerate(sparkline_values):
            x = pad + i * (w - 2 * pad) / (n - 1)
            y = h - pad - (v - vmin) / rng * (h - 2 * pad)
            points.append(f"{x:.1f},{y:.1f}")
        path_d = "M " + " L ".join(points)
        area_d = path_d + f" L {points[-1].split(',')[0]},{h} L {points[0].split(',')[0]},{h} Z"
        sparkline_svg = (
            f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="none" '
            f'style="width:100%;height:30px;display:block;margin-top:6px;">'
            f'<path d="{area_d}" fill="{sparkline_color}" opacity="0.15"/>'
            f'<path d="{path_d}" stroke="{sparkline_color}" stroke-width="1.5" fill="none"/>'
            f'</svg>'
        )
    else:
        sparkline_svg = ""

    delta_html = ""
    if delta:
        if delta_positive is True:
            d_color = COLORS["trend_up_good"]; arrow = icon("trending-up", size=11, color=d_color)
        elif delta_positive is False:
            d_color = COLORS["trend_down_bad"]; arrow = icon("trending-down", size=11, color=d_color)
        else:
            d_color = COLORS["trend_neutral"]; arrow = icon("minus", size=11, color=d_color)
        delta_html = (
            f'<div style="display:inline-flex;align-items:center;gap:4px;color:{d_color};'
            f'font-size:0.72rem;font-weight:600;margin-top:4px;">{arrow}<span>{delta}</span></div>'
        )

    html = (
        f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};'
        f'border-radius:8px;padding:1rem 1.15rem;">'
        f'<div style="color:{COLORS["text_muted"]};font-size:0.68rem;font-weight:600;'
        f'text-transform:uppercase;letter-spacing:0.07em;margin-bottom:8px;">{label}</div>'
        f'<div style="display:flex;align-items:baseline;justify-content:space-between;gap:8px;">'
        f'<div style="color:{COLORS["text"]};font-size:1.5rem;font-weight:700;font-family:Inter,sans-serif;line-height:1.1;letter-spacing:-0.02em;">{value}</div>'
        f'{delta_html}</div>'
        f'{sparkline_svg}</div>'
    )
    _render_html(html)


def kpi(label: str, value: str, delta: Optional[str] = None, delta_positive: Optional[bool] = None, icon_name: Optional[str] = None):
    is_trend = False
    if delta:
        d = delta.strip()
        if d.startswith(("+", "-")) and any(c.isdigit() for c in d):
            is_trend = True
    if is_trend:
        metric_card(label, value, delta=delta, delta_positive=delta_positive)
    else:
        metric_card(label, value, sublabel=delta)


def kpi_row(items: list):
    if not items:
        return
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        with col:
            metric_card(
                label=item.get("label", ""), value=item.get("value", ""),
                delta=item.get("delta"), delta_positive=item.get("delta_positive"),
                sublabel=item.get("sublabel"),
            )


def info_card(title: str, body: str, icon_name: Optional[str] = None):
    icon_html = f'<div style="margin-bottom:8px;">{icon(icon_name, size=18, color=COLORS["text_muted"])}</div>' if icon_name else ""
    html = (
        f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};border-radius:8px;padding:1rem 1.15rem;">'
        f'{icon_html}'
        f'<div style="color:{COLORS["text"]};font-size:0.92rem;font-weight:600;margin-bottom:6px;">{title}</div>'
        f'<div style="color:{COLORS["text_muted"]};font-size:0.84rem;line-height:1.5;">{body}</div>'
        f'</div>'
    )
    _render_html(html)


def empty_state(message: str, hint: Optional[str] = None):
    hint_html = f'<div style="color:{COLORS["text_dim"]};font-size:0.8rem;margin-top:8px;">{hint}</div>' if hint else ""
    html = (
        f'<div style="background:{COLORS["surface"]};border:1px dashed {COLORS["border_strong"]};border-radius:8px;padding:2.5rem 1.5rem;text-align:center;">'
        f'<div style="color:{COLORS["text_muted"]};font-size:0.88rem;font-weight:500;">{message}</div>'
        f'{hint_html}</div>'
    )
    _render_html(html)


def alert_card(title: str = "", severity: str = "medium", body: str = "", timestamp: Optional[str] = None,
               score: Optional[float] = None, txn_id: Optional[str] = None, amount: Optional[str] = None):
    sev_lower = severity.lower()
    sev_map = {"critical": "critical", "high": "high", "medium": "medium", "low": "low"}
    color_key = sev_map.get(sev_lower, "info")
    color = COLORS.get(color_key, COLORS["info"])
    pill = status_pill(severity.upper(), color_key)
    score_html = f'<span style="color:{COLORS["text_muted"]};font-family:JetBrains Mono,monospace;font-size:0.76rem;font-weight:500;margin-left:8px;">{score*100:.1f}%</span>' if score is not None else ""
    right_html = ""
    if timestamp:
        right_html = f'<span style="color:{COLORS["text_dim"]};font-size:0.74rem;font-family:JetBrains Mono,monospace;">{timestamp}</span>'
    elif amount:
        right_html = f'<span style="color:{COLORS["text"]};font-feature-settings:tnum;font-size:0.84rem;font-weight:600;">{amount}</span>'
    body_html = f'<div style="color:{COLORS["text_muted"]};font-size:0.8rem;line-height:1.5;margin-top:6px;">{body}</div>' if body else ""
    if txn_id and not title:
        title = f"Txn {txn_id}"
    html = (
        f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};border-left:3px solid {color};border-radius:6px;padding:0.7rem 0.95rem;margin-bottom:0.5rem;">'
        f'<div style="display:flex;align-items:center;justify-content:space-between;gap:10px;">'
        f'<div style="display:flex;align-items:center;gap:10px;flex:1;min-width:0;">'
        f'{pill}'
        f'<span style="color:{COLORS["text"]};font-size:0.86rem;font-weight:500;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{title}</span>'
        f'{score_html}'
        f'</div>{right_html}</div>{body_html}</div>'
    )
    _render_html(html)


def stage_indicator(name: str, status: str = "pending") -> str:
    status_color = {"pending": COLORS["text_dim"], "running": COLORS["medium"], "done": COLORS["low"], "error": COLORS["critical"]}.get(status, COLORS["text_dim"])
    icon_name = {"pending": "clock", "running": "refresh-cw", "done": "check-circle", "error": "x-circle"}.get(status, "clock")
    icon_svg = icon(icon_name, size=14, color=status_color)
    return (
        f'<span style="display:inline-flex;align-items:center;gap:6px;color:{status_color};font-size:0.8rem;font-weight:600;">'
        f'{icon_svg}{name}</span>'
    )


def divider():
    _render_html(f'<hr style="border:none;border-top:1px solid {COLORS["border"]};margin:1.4rem 0;">')


def render_top_bar():
    """Minimal top bar: user chip + BYOK status + settings + sign out on the right.

    The BYOK badge is rendered inline below the user chip. It always reflects
    the current key-source state ('session' | 'env' | None) and links to
    Settings. byok is imported lazily because byok itself imports from
    components — without lazy import this would be circular.
    """
    col_spacer, col_user = st.columns([6, 2])

    with col_user:
        email = st.session_state.get("user_email", "")
        if email:
            initials = "".join(p[0] for p in email.split("@")[0].split(".") if p)[:2].upper() or email[:2].upper()
            _render_html(
                f'<div style="display:flex;justify-content:flex-end;align-items:center;gap:8px;">'
                f'<div style="background:{COLORS["accent"]};width:30px;height:30px;border-radius:50%;display:flex;align-items:center;justify-content:center;color:#0e1117;font-weight:700;font-size:0.76rem;">{initials}</div>'
                f'<span style="color:{COLORS["text"]};font-size:0.82rem;font-weight:500;">{email.split("@")[0]}</span>'
                f'</div>'
            )

            # BYOK connection-status badge — always visible, links to Settings.
            try:
                from src.dashboard.byok import render_byok_sidebar_badge
                _render_html('<div style="height:0.5rem;"></div>')
                render_byok_sidebar_badge()
            except Exception:
                # If byok module isn't ready yet (e.g. on first load before file
                # is created), fail silently rather than break the top bar.
                pass

            uc1, uc2 = st.columns([1, 1])
            with uc1:
                if st.button("Settings", key="_top_settings_btn", use_container_width=True):
                    st.switch_page("pages/7_Settings.py")
            with uc2:
                from src.dashboard import auth as supabase_auth
                if st.button("Sign out", key="_top_signout_btn", use_container_width=True):
                    supabase_auth.sign_out()
                    st.rerun()

    _render_html(f'<div style="border-bottom:1px solid {COLORS["border"]};margin:0.6rem 0 1.2rem 0;"></div>')


def _mark_google_button(key: str):
    js = (
        f'<script>'
        f'(function() {{'
        f'  const buttons = window.parent.document.querySelectorAll(\'div[data-testid="stButton"] button\');'
        f'  buttons.forEach(b => {{ if (b.innerText.includes("Google")) {{ b.setAttribute("data-google", "1"); }} }});'
        f'}})();'
        f'</script>'
    )
    st.markdown(js, unsafe_allow_html=True)


def _login_form():
    from src.dashboard import auth as supabase_auth

    _render_html(
        f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};border-radius:12px;padding:1.8rem 2rem;margin-bottom:1rem;">'
        f'<div style="color:{COLORS["text"]};font-size:1.15rem;font-weight:600;margin-bottom:0.3rem;letter-spacing:-0.01em;">Sign in to your account</div>'
        f'<div style="color:{COLORS["text_muted"]};font-size:0.85rem;">Welcome back. Please enter your details.</div>'
        f'</div>'
    )

    if st.button("Continue with Google", use_container_width=True, key="google_signin", type="secondary"):
        url = supabase_auth.sign_in_with_google()
        if url:
            st.markdown(f'<meta http-equiv="refresh" content="0; url={url}">', unsafe_allow_html=True)
            st.markdown(f'<a href="{url}" target="_self" style="color:{COLORS["accent"]};">If not redirected, click here</a>', unsafe_allow_html=True)
            st.stop()
    _mark_google_button("google_signin")

    _render_html(
        f'<div style="display:flex;align-items:center;gap:12px;margin:1.2rem 0;">'
        f'<div style="flex:1;height:1px;background:{COLORS["border"]};"></div>'
        f'<span style="color:{COLORS["text_dim"]};font-size:0.78rem;text-transform:uppercase;letter-spacing:0.06em;">or</span>'
        f'<div style="flex:1;height:1px;background:{COLORS["border"]};"></div>'
        f'</div>'
    )

    _render_html(f'<div style="color:{COLORS["text"]};font-size:0.82rem;font-weight:500;margin-bottom:6px;">Email</div>')
    email = st.text_input("Email", key="login_email", label_visibility="collapsed", placeholder="name@company.com")

    _render_html(f'<div style="color:{COLORS["text"]};font-size:0.82rem;font-weight:500;margin-bottom:6px;margin-top:0.8rem;">Password</div>')
    password = st.text_input("Password", type="password", key="login_pw", label_visibility="collapsed", placeholder="••••••••")

    fp_col1, fp_col2 = st.columns([3, 1])
    with fp_col2:
        if st.button("Forgot?", key="forgot_pw_btn", use_container_width=True):
            st.session_state["auth_view"] = "forgot"
            st.rerun()

    sign_in = st.button("Sign in", use_container_width=True, key="login_signin_btn")

    su_col1, su_col2 = st.columns([3, 1])
    with su_col1:
        _render_html(
            f'<div style="text-align:right;margin-top:1rem;color:{COLORS["text_muted"]};font-size:0.82rem;padding-top:8px;">'
            f"Don't have an account?"
            f'</div>'
        )
    with su_col2:
        if st.button("Sign up", key="goto_signup_btn", use_container_width=True):
            st.session_state["auth_view"] = "signup"
            st.rerun()

    if sign_in:
        if not email or not password:
            st.error("Please enter your email and password")
        else:
            with st.spinner("Signing you in..."):
                success, msg = supabase_auth.sign_in_with_password(email, password)
            if success:
                st.rerun()
            else:
                st.error(msg)


def _signup_form():
    from src.dashboard import auth as supabase_auth

    _render_html(
        f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};border-radius:12px;padding:1.8rem 2rem;margin-bottom:1rem;">'
        f'<div style="color:{COLORS["text"]};font-size:1.15rem;font-weight:600;margin-bottom:0.3rem;letter-spacing:-0.01em;">Create your account</div>'
        f'<div style="color:{COLORS["text_muted"]};font-size:0.85rem;">Get started with FraudSentinel in seconds.</div>'
        f'</div>'
    )

    if st.button("Sign up with Google", use_container_width=True, key="google_signup", type="secondary"):
        url = supabase_auth.sign_in_with_google()
        if url:
            st.markdown(f'<meta http-equiv="refresh" content="0; url={url}">', unsafe_allow_html=True)
            st.markdown(f'<a href="{url}" target="_self" style="color:{COLORS["accent"]};">If not redirected, click here</a>', unsafe_allow_html=True)
            st.stop()
    _mark_google_button("google_signup")

    _render_html(
        f'<div style="display:flex;align-items:center;gap:12px;margin:1.2rem 0;">'
        f'<div style="flex:1;height:1px;background:{COLORS["border"]};"></div>'
        f'<span style="color:{COLORS["text_dim"]};font-size:0.78rem;text-transform:uppercase;letter-spacing:0.06em;">or</span>'
        f'<div style="flex:1;height:1px;background:{COLORS["border"]};"></div>'
        f'</div>'
    )

    _render_html(f'<div style="color:{COLORS["text"]};font-size:0.82rem;font-weight:500;margin-bottom:6px;">Full name</div>')
    name = st.text_input("Name", key="signup_name", label_visibility="collapsed", placeholder="Jane Doe")

    _render_html(f'<div style="color:{COLORS["text"]};font-size:0.82rem;font-weight:500;margin-bottom:6px;margin-top:0.8rem;">Work email</div>')
    email = st.text_input("Email", key="signup_email", label_visibility="collapsed", placeholder="jane@company.com")

    _render_html(f'<div style="color:{COLORS["text"]};font-size:0.82rem;font-weight:500;margin-bottom:6px;margin-top:0.8rem;">Password</div>')
    password = st.text_input("Password", type="password", key="signup_pw", label_visibility="collapsed", placeholder="Min. 6 characters")

    _render_html(
        f'<div style="color:{COLORS["text_dim"]};font-size:0.74rem;line-height:1.5;margin:0.8rem 0 1.1rem 0;">'
        f'By creating an account, you agree to our Terms of Service and Privacy Policy.'
        f'</div>'
    )

    create = st.button("Create account", use_container_width=True, key="signup_create_btn")

    si_col1, si_col2 = st.columns([3, 1])
    with si_col1:
        _render_html(
            f'<div style="text-align:right;margin-top:1rem;color:{COLORS["text_muted"]};font-size:0.82rem;padding-top:8px;">'
            f'Already have an account?'
            f'</div>'
        )
    with si_col2:
        if st.button("Sign in", key="goto_signin_btn", use_container_width=True):
            st.session_state["auth_view"] = "login"
            st.rerun()

    if create:
        if not email or not password:
            st.error("Please fill in all fields")
        elif len(password) < 6:
            st.error("Password must be at least 6 characters")
        else:
            with st.spinner("Creating your account..."):
                success, msg = supabase_auth.sign_up_with_password(email, password, name)
            if success:
                if "signed in" in msg.lower():
                    st.rerun()
                else:
                    st.success(msg)
            else:
                st.error(msg)


def _forgot_form():
    from src.dashboard import auth as supabase_auth

    _render_html(
        f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};border-radius:12px;padding:1.8rem 2rem;margin-bottom:1rem;">'
        f'<div style="color:{COLORS["text"]};font-size:1.15rem;font-weight:600;margin-bottom:0.3rem;letter-spacing:-0.01em;">Reset your password</div>'
        f'<div style="color:{COLORS["text_muted"]};font-size:0.85rem;">We\'ll email you a link to reset your password.</div>'
        f'</div>'
    )

    _render_html(f'<div style="color:{COLORS["text"]};font-size:0.82rem;font-weight:500;margin-bottom:6px;">Email</div>')
    email = st.text_input("Email", key="forgot_email", label_visibility="collapsed", placeholder="name@company.com")

    _render_html('<div style="height:14px;"></div>')

    send = st.button("Send reset link", use_container_width=True, key="forgot_send_btn")

    bk_col1, bk_col2 = st.columns([3, 1])
    with bk_col2:
        if st.button("Back", key="forgot_back_btn", use_container_width=True):
            st.session_state["auth_view"] = "login"
            st.rerun()

    if send:
        if not email:
            st.error("Please enter your email")
        else:
            with st.spinner("Sending reset link..."):
                success, msg = supabase_auth.send_password_reset(email)
            if success:
                st.success(msg)
            else:
                st.error(msg)


def render_login_gate() -> bool:
    from src.dashboard import auth as supabase_auth

    supabase_auth.handle_oauth_callback()

    if st.session_state.get("logged_in"):
        return True

    apply_theme()

    if not supabase_auth.is_supabase_configured():
        st.error("Authentication not configured. Set SUPABASE_URL and SUPABASE_ANON_KEY in .env")
        st.stop()

    st.markdown(
        '<style>[data-testid="stSidebar"] { display: none !important; } '
        '[data-testid="collapsedControl"] { display: none !important; }</style>',
        unsafe_allow_html=True,
    )

    view = st.session_state.get("auth_view", "login")

    _render_html('<div style="height:40px;"></div>')

    pad_l, form_col, pad_r = st.columns([1, 2, 1])
    with form_col:
        _render_html(
            f'<div style="display:flex;flex-direction:column;align-items:center;gap:14px;margin-bottom:1.8rem;">'
            f'<div style="background:{COLORS["accent"]};width:46px;height:46px;border-radius:11px;display:flex;align-items:center;justify-content:center;box-shadow:0 6px 24px rgba(94,234,212,0.25);">'
            f'{icon("shield", size=24, color="#0e1117")}</div>'
            f'<div style="color:{COLORS["text"]};font-size:1.5rem;font-weight:700;letter-spacing:-0.02em;">FraudSentinel</div>'
            f'</div>'
        )

        if view == "login":
            _login_form()
        elif view == "signup":
            _signup_form()
        elif view == "forgot":
            _forgot_form()

    return False


def render_user_chip():
    email = st.session_state.get("user_email", "")
    role = st.session_state.get("user_role", "Fraud Analyst")
    if not email:
        return
    initials = "".join(p[0] for p in email.split("@")[0].split(".") if p)[:2].upper() or email[:2].upper()
    _render_html(
        f'<div style="display:flex;align-items:center;gap:10px;padding:9px 11px;background:{COLORS["surface_2"]};border:1px solid {COLORS["border"]};border-radius:8px;margin-bottom:0.7rem;">'
        f'<div style="background:{COLORS["accent"]};width:28px;height:28px;border-radius:50%;display:flex;align-items:center;justify-content:center;color:#0e1117;font-weight:700;font-size:0.74rem;">{initials}</div>'
        f'<div style="flex:1;min-width:0;">'
        f'<div style="color:{COLORS["text"]};font-size:0.8rem;font-weight:500;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{email}</div>'
        f'<div style="color:{COLORS["text_muted"]};font-size:0.7rem;">{role}</div>'
        f'</div></div>'
    )


def style_plotly(fig, height: int = 320):
    fig.update_layout(
        height=height,
        margin=dict(l=12, r=12, t=20, b=12),
        plot_bgcolor=CHART_BG,
        paper_bgcolor=CHART_BG,
        font=dict(color=CHART_TEXT, family="Inter, sans-serif", size=11),
        xaxis=dict(gridcolor=CHART_GRID, zeroline=False, showline=False, color=CHART_TEXT),
        yaxis=dict(gridcolor=CHART_GRID, zeroline=False, showline=False, color=CHART_TEXT),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=CHART_GRID, borderwidth=1, font=dict(size=10)),
        hoverlabel=dict(bgcolor=COLORS["surface_2"], font=dict(family="Inter, sans-serif", size=11, color=COLORS["text"]), bordercolor=COLORS["border"]),
    )
    return fig


def transactions_table(df, max_rows: int = 50, score_col: str = "xgb_score"):
    if df is None or len(df) == 0:
        empty_state("No transactions match current filters")
        return

    df = df.head(max_rows).copy()
    rows = []
    for _, row in df.iterrows():
        score = float(row.get(score_col, 0))
        label, color = risk_band(score)
        amt = float(row.get("TransactionAmt", 0))
        product = str(row.get("ProductCD", "?"))
        vel = int(row.get("card1_txn_count_24h", 0))
        z = float(row.get("card1_amt_zscore", 0))
        txn_id = str(row.get("TransactionID", ""))
        card_id = str(row.get("card1", ""))

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
            f'<td style="padding:9px 12px;color:{COLORS["text_muted"]};font-size:0.84rem;text-align:right;">{vel}</td>'
            f'<td style="padding:9px 12px;color:{COLORS["text_muted"]};font-family:JetBrains Mono,monospace;font-size:0.8rem;text-align:right;">{z:+.2f}\u03c3</td>'
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
        f'<th style="{head_cell}text-align:right;">24h count</th>'
        f'<th style="{head_cell}text-align:right;">Anomaly</th>'
        f'</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody>'
        f'</table></div>'
    )
    _render_html(table_html)


def detail_grid(items: list):
    rows = []
    for item in items:
        lbl = item.get("label", "")
        val = item.get("value", "")
        mono = item.get("mono", False)
        val_style = f"color:{COLORS['text']};font-size:0.85rem;"
        if mono:
            val_style += "font-family:JetBrains Mono,monospace;"
        rows.append(
            f'<div style="display:flex;justify-content:space-between;align-items:center;padding:9px 14px;border-bottom:1px solid {COLORS["border"]};">'
            f'<span style="color:{COLORS["text_muted"]};font-size:0.78rem;">{lbl}</span>'
            f'<span style="{val_style}">{val}</span>'
            f'</div>'
        )
    html = f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};border-radius:8px;overflow:hidden;">{"".join(rows)}</div>'
    _render_html(html)
    