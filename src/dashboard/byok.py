"""Bring-your-own-key (BYOK) helpers for Anthropic API access.

Strict BYOK design:
  - Visitor's API key lives ONLY in st.session_state (never persisted to disk
    or database; clears when tab closes)
  - Every LLM-consuming surface gates behind require_api_key() before calling
    Anthropic
  - Local development still works via ANTHROPIC_API_KEY env var as fallback —
    so the developer doesn't have to paste their key on every refresh
  - Single canonical input point: the Settings page. Other pages link there
    when a key is missing.

Public API:
  get_api_key()              → str | None
  get_api_key_source()       → "session" | "env" | None
  has_api_key()              → bool
  require_api_key(...)       → str | None  (renders gate if no key, returns None)
  render_byok_sidebar_badge() → None  (status badge for top bar)
"""

from __future__ import annotations

import os

import streamlit as st

from src.dashboard.components import COLORS, _render_html, icon

# Must match the key used by Settings page for storage continuity
SESSION_KEY = "anthropic_api_key"


def get_api_key() -> str | None:
    """Return the active API key, preferring session over env. None if neither."""
    sess = st.session_state.get(SESSION_KEY)
    if sess and isinstance(sess, str) and sess.strip():
        return sess.strip()
    env = os.environ.get("ANTHROPIC_API_KEY")
    if env and isinstance(env, str) and env.strip():
        return env.strip()
    return None


def get_api_key_source() -> str | None:
    """Returns 'session' if user-provided, 'env' if developer fallback, else None."""
    sess = st.session_state.get(SESSION_KEY)
    if sess and isinstance(sess, str) and sess.strip():
        return "session"
    env = os.environ.get("ANTHROPIC_API_KEY")
    if env and isinstance(env, str) and env.strip():
        return "env"
    return None


def has_api_key() -> bool:
    return get_api_key() is not None


def require_api_key(
    action_label: str,
    description: str | None = None,
    estimated_cost: str | None = None,
) -> str | None:
    """Return the key, or render a 'Connect in Settings' gate and return None."""
    key = get_api_key()
    if key:
        return key

    info_bg = COLORS.get("info_bg", COLORS["surface_2"])

    cost_html = ""
    if estimated_cost:
        cost_html = (
            f'<div style="display:flex;align-items:center;gap:6px;margin-top:8px;'
            f'color:{COLORS["text_dim"]};font-size:0.76rem;">'
            f'{icon("dollar-sign", size=12, color=COLORS["text_dim"])}'
            f'<span>Estimated cost: <span style="color:{COLORS["text_muted"]};font-family:JetBrains Mono,monospace;">{estimated_cost}</span></span>'
            f'</div>'
        )

    desc_html = ""
    if description:
        desc_html = (
            f'<div style="color:{COLORS["text_muted"]};font-size:0.84rem;line-height:1.55;margin-bottom:10px;">'
            f'{description}'
            f'</div>'
        )

    _render_html(
        f'<div style="background:{info_bg};border:1px solid {COLORS["info"]};border-left:3px solid {COLORS["info"]};'
        f'border-radius:8px;padding:1.1rem 1.3rem;margin-bottom:1rem;">'
        f'<div style="display:flex;align-items:flex-start;gap:12px;">'
        f'<div style="flex-shrink:0;margin-top:2px;">'
        f'{icon("key", size=18, color=COLORS["info"])}'
        f'</div>'
        f'<div style="flex:1;">'
        f'<div style="color:{COLORS["text"]};font-size:0.95rem;font-weight:600;margin-bottom:5px;">'
        f'Connect your Anthropic API key to continue'
        f'</div>'
        f'<div style="color:{COLORS["text_muted"]};font-size:0.84rem;line-height:1.55;margin-bottom:10px;">'
        f'<strong style="color:{COLORS["text"]};">{action_label}</strong> uses Anthropic\'s Claude API. '
        f'This demo uses bring-your-own-key — your key stays in this browser session only and is never saved.'
        f'</div>'
        f'{desc_html}'
        f'<div style="display:flex;align-items:center;gap:14px;margin-top:12px;flex-wrap:wrap;">'
        f'<a href="/Settings" target="_self" style="text-decoration:none;">'
        f'<span style="background:{COLORS["accent"]};color:#0e1117;padding:7px 14px;border-radius:6px;'
        f'font-size:0.84rem;font-weight:600;display:inline-flex;align-items:center;gap:6px;">'
        f'Go to Settings'
        f'<span style="font-size:0.95rem;line-height:1;">→</span>'
        f'</span>'
        f'</a>'
        f'<a href="https://console.anthropic.com" target="_blank" rel="noopener" '
        f'style="color:{COLORS["text_muted"]};font-size:0.82rem;text-decoration:none;display:inline-flex;align-items:center;gap:5px;">'
        f'Get a key from console.anthropic.com'
        f'{icon("external-link", size=12, color=COLORS["text_muted"])}'
        f'</a>'
        f'</div>'
        f'{cost_html}'
        f'</div>'
        f'</div>'
        f'</div>'
    )

    return None


def render_byok_sidebar_badge() -> None:
    """Status badge for the top bar — always visible, links to Settings."""
    source = get_api_key_source()

    if source == "session":
        bg = COLORS.get("low_bg", COLORS["surface_2"])
        border = "rgba(95,201,164,0.3)"
        ic_name = "check-circle"
        ic_color = COLORS["low"]
        text_color = COLORS["low"]
        label = "Anthropic key connected"
        sublabel = "Session only"
    elif source == "env":
        bg = COLORS["surface_2"]
        border = COLORS["border"]
        ic_name = "key"
        ic_color = COLORS["text_muted"]
        text_color = COLORS["text_muted"]
        label = "Anthropic key (dev env)"
        sublabel = "From .env"
    else:
        bg = COLORS["surface_2"]
        border = COLORS["border"]
        ic_name = "key"
        ic_color = COLORS["text_dim"]
        text_color = COLORS["text_dim"]
        label = "No Anthropic key"
        sublabel = "Connect in Settings"

    _render_html(
        f'<a href="/Settings" target="_self" style="text-decoration:none;display:block;">'
        f'<div style="background:{bg};border:1px solid {border};border-radius:6px;'
        f'padding:7px 10px;display:flex;align-items:center;gap:8px;margin-bottom:0.5rem;'
        f'cursor:pointer;transition:background 0.15s;">'
        f'<div style="flex-shrink:0;">{icon(ic_name, size=13, color=ic_color)}</div>'
        f'<div style="flex:1;min-width:0;">'
        f'<div style="color:{text_color};font-size:0.76rem;font-weight:500;line-height:1.2;">{label}</div>'
        f'<div style="color:{COLORS["text_dim"]};font-size:0.68rem;line-height:1.2;margin-top:1px;">{sublabel}</div>'
        f'</div>'
        f'</div>'
        f'</a>'
    )
    