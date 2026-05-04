"""Pattern Library — fraud patterns the AI uses to find similar cases.

Each indicator now renders with the plain-English explanation as the headline
and the technical threshold as monospace subtext below — best of both worlds
for the analyst (instant scan + verifiable evidence).
"""

import json

import streamlit as st

st.set_page_config(page_title="Pattern Library — FraudSentinel", layout="wide")

from src.dashboard.components import (  # noqa: E402
    COLORS, _render_html, apply_theme, empty_state, icon,
    page_header, render_login_gate, render_top_bar, section_header,
)
from src.utils.config import FRAUD_CASES_DIR  # noqa: E402

apply_theme()
if not render_login_gate():
    st.stop()

render_top_bar()

page_header(
    "Pattern Library",
    "A catalog of fraud patterns your AI agents reference when investigating new cases.",
)


@st.cache_data
def load_cases():
    cases = []
    if not FRAUD_CASES_DIR.exists():
        return cases
    for path in sorted(FRAUD_CASES_DIR.glob("case_*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                cases.append(json.load(f))
        except Exception:
            continue
    return cases


cases = load_cases()
if not cases:
    empty_state(
        "Your pattern library is empty",
        "Run `python -m src.agentic.generate_patterns_v2` followed by `python -m src.agentic.build_knowledge_base` to populate it.",
    )
    st.stop()


PATTERN_COLORS = {
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
    # New archetypes
    "device_takeover": "#7dd3fc",         # sky blue
    "credential_compromise": "#fbbf24",   # amber
    "engineered_anomaly": "#c084fc",      # purple
}

PATTERN_DESCRIPTIONS = {
    "card_testing": "Small charges to validate stolen cards before larger purchases",
    "geo_anomaly": "Transactions from unexpected geographic locations",
    "account_takeover": "Unauthorized access to legitimate customer accounts",
    "velocity_attack": "Rapid succession of transactions on the same card",
    "synthetic_identity": "Fabricated identities mixing real and fake details",
    "bin_attack": "Probing card BIN ranges to find valid numbers",
    "friendly_fraud": "Legitimate purchase later disputed by the cardholder",
    "temporal_anomaly": "Transactions at unusual times for the card's pattern",
    "email_risk": "Disposable or recently created email signals",
    "subscription_probe": "Testing card validity through subscription signups",
    # New archetypes
    "device_takeover": "Stolen credentials used from a never-seen device",
    "credential_compromise": "Valid credentials misused at unusual aggregated rate",
    "engineered_anomaly": "Composite Vesta features flag what raw signals miss",
}


# ====================================================================
# Category overview
# ====================================================================
patterns = sorted(set(c["pattern"] for c in cases))

category_counts = {}
for c in cases:
    p = c["pattern"]
    category_counts[p] = category_counts.get(p, 0) + 1

sorted_cats = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)

section_header(
    "What's in the library",
    f"{len(cases):,} fraud patterns across {len(patterns)} categories",
)

cards_html = ""
for cat, count in sorted_cats:
    color = PATTERN_COLORS.get(cat, COLORS["text_muted"])
    label = cat.replace("_", " ").title()
    desc = PATTERN_DESCRIPTIONS.get(cat, "")

    cards_html += (
        f'<div style="background:{color};border-radius:10px;padding:1.2rem 1.3rem;'
        f'display:flex;flex-direction:column;justify-content:space-between;'
        f'min-height:170px;cursor:default;color:#0e1117;">'
        f'<div>'
        f'<div style="font-size:0.95rem;font-weight:700;letter-spacing:-0.01em;margin-bottom:6px;">{label}</div>'
        f'<div style="font-size:0.78rem;line-height:1.45;opacity:0.78;font-weight:500;">{desc}</div>'
        f'</div>'
        f'<div style="font-size:1.8rem;font-weight:800;font-family:Inter,sans-serif;letter-spacing:-0.02em;margin-top:10px;">'
        f'{count}<span style="font-size:0.78rem;font-weight:600;opacity:0.7;margin-left:6px;">patterns</span>'
        f'</div>'
        f'</div>'
    )

# Adapt grid columns to category count — 5 cols if many categories, fewer if not
n_cats = len(sorted_cats)
n_cols = 5 if n_cats >= 5 else max(2, n_cats)

grid_html = (
    f'<div style="display:grid;grid-template-columns:repeat({n_cols},1fr);gap:12px;margin:0.8rem 0 1.6rem 0;">'
    f'{cards_html}'
    f'</div>'
)
_render_html(grid_html)


# ====================================================================
# Search & filter
# ====================================================================
section_header("Browse patterns")

search_col, filter_col, page_size_col = st.columns([3, 1.5, 1])
with search_col:
    query = st.text_input(
        "Search patterns",
        placeholder="Search by pattern name, behavior, or indicator…",
        label_visibility="collapsed",
        key="pl_search",
    )

with filter_col:
    pattern_filter = st.selectbox(
        "Pattern category",
        ["All categories"] + patterns,
        label_visibility="collapsed",
        key="pl_filter",
    )

with page_size_col:
    page_size = st.selectbox(
        "Per page",
        [10, 25, 50, 100],
        index=1,
        label_visibility="collapsed",
        key="pl_page_size",
    )


search_state_key = f"{query}::{pattern_filter}::{page_size}"
if st.session_state.get("pl_last_search") != search_state_key:
    st.session_state["pl_page"] = 1
    st.session_state["pl_last_search"] = search_state_key


filtered = cases
if query:
    q = query.lower()
    filtered = [
        c for c in filtered
        if q in c["title"].lower()
        or q in c["narrative"].lower()
        or q in c["pattern"].lower()
        or any(q in i.lower() for i in c.get("indicators", []))
        or any(q in e.lower() for e in c.get("indicator_explanations", []))
    ]

if pattern_filter != "All categories":
    filtered = [c for c in filtered if c["pattern"] == pattern_filter]


# ====================================================================
# Pagination
# ====================================================================
total_filtered = len(filtered)
total_pages = max(1, (total_filtered + page_size - 1) // page_size)

if "pl_page" not in st.session_state:
    st.session_state["pl_page"] = 1
current_page = min(max(1, st.session_state["pl_page"]), total_pages)
st.session_state["pl_page"] = current_page

start_idx = (current_page - 1) * page_size
end_idx = min(start_idx + page_size, total_filtered)
page_cases = filtered[start_idx:end_idx]


_render_html(
    f'<div style="display:flex;justify-content:space-between;align-items:center;margin:1rem 0 1.2rem 0;">'
    f'<div style="color:{COLORS["text_muted"]};font-size:0.82rem;">'
    f'Showing <span style="color:{COLORS["text"]};font-weight:500;font-family:JetBrains Mono,monospace;">{start_idx + 1 if total_filtered > 0 else 0}–{end_idx}</span> '
    f'of <span style="color:{COLORS["text"]};font-weight:500;font-family:JetBrains Mono,monospace;">{total_filtered:,}</span> patterns'
    f'</div>'
    f'<div style="color:{COLORS["text_muted"]};font-size:0.82rem;">'
    f'Page <span style="color:{COLORS["text"]};font-weight:500;">{current_page}</span> of {total_pages}'
    f'</div>'
    f'</div>'
)


# ====================================================================
# Pattern cards
# ====================================================================
def _render_indicator_row(explanation: str | None, technical: str) -> str:
    """Produce HTML for one indicator row.

    Plain-English headline on top, technical threshold as monospace subtext.
    Falls back to technical-only if explanation is missing.
    """
    if explanation:
        return (
            f'<div style="background:{COLORS["surface_2"]};border:1px solid {COLORS["border"]};'
            f'border-radius:6px;padding:0.7rem 0.95rem;margin-bottom:0.45rem;">'
            f'<div style="color:{COLORS["text"]};font-size:0.86rem;line-height:1.5;font-weight:500;">'
            f'{explanation}'
            f'</div>'
            f'<div style="color:{COLORS["text_dim"]};font-family:JetBrains Mono,monospace;font-size:0.72rem;'
            f'margin-top:5px;line-height:1.5;letter-spacing:-0.005em;">'
            f'{technical}'
            f'</div>'
            f'</div>'
        )
    else:
        return (
            f'<div style="background:{COLORS["surface_2"]};border:1px solid {COLORS["border"]};color:{COLORS["text_muted"]};'
            f'padding:0.55rem 0.85rem;border-radius:5px;font-size:0.78rem;font-family:JetBrains Mono,monospace;'
            f'margin-bottom:0.4rem;line-height:1.5;">'
            f'{technical}'
            f'</div>'
        )


if not page_cases:
    empty_state(
        "No patterns match your search",
        "Try a different keyword or clear the filter to see all patterns.",
    )
else:
    for case in page_cases:
        color = PATTERN_COLORS.get(case["pattern"], COLORS["text_muted"])
        pattern_label = case["pattern"].replace("_", " ").title()

        indicators = case.get("indicators", []) or []
        explanations = case.get("indicator_explanations", []) or []

        # Render paired rows
        indicator_rows_html = ""
        for i, ind in enumerate(indicators):
            explanation = explanations[i] if i < len(explanations) else None
            indicator_rows_html += _render_indicator_row(explanation, ind)

        _render_html(
            f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};border-left:3px solid {color};'
            f'padding:1.1rem 1.3rem;border-radius:8px;margin-bottom:0.8rem;">'
            f'<div style="display:flex;justify-content:space-between;align-items:flex-start;gap:14px;margin-bottom:10px;">'
            f'<div style="color:{COLORS["text"]};font-size:0.98rem;font-weight:600;letter-spacing:-0.01em;">{case["title"]}</div>'
            f'<div style="display:flex;gap:14px;align-items:center;flex-shrink:0;">'
            f'<span style="background:{color};color:#0e1117;padding:3px 9px;border-radius:4px;font-size:0.68rem;font-weight:700;'
            f'text-transform:uppercase;letter-spacing:0.06em;">{pattern_label}</span>'
            f'<span style="color:{COLORS["text_dim"]};font-size:0.76rem;font-family:JetBrains Mono,monospace;">{case["id"]}</span>'
            f'</div>'
            f'</div>'
            f'<div style="color:{COLORS["text_muted"]};font-size:0.86rem;line-height:1.6;margin-bottom:14px;">{case["narrative"]}</div>'
            f'<div style="margin-top:6px;">'
            f'<div style="color:{COLORS["text_dim"]};font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:8px;">Behavioral indicators</div>'
            f'{indicator_rows_html}'
            f'</div>'
            f'</div>'
        )


# ====================================================================
# Pagination controls
# ====================================================================
if total_pages > 1:
    _render_html('<div style="height:0.6rem;"></div>')

    def get_page_numbers(current, total, window=2):
        pages = set()
        for i in range(1, min(3, total + 1)):
            pages.add(i)
        for i in range(max(1, total - 1), total + 1):
            pages.add(i)
        for i in range(max(1, current - window), min(total + 1, current + window + 1)):
            pages.add(i)
        return sorted(pages)

    page_nums = get_page_numbers(current_page, total_pages)

    nav_cols = st.columns([1] + [0.6] * len(page_nums) + [1])

    with nav_cols[0]:
        if st.button("← Prev", key="pl_prev", disabled=(current_page == 1), use_container_width=True):
            st.session_state["pl_page"] = max(1, current_page - 1)
            st.rerun()

    prev_p = 0
    for i, p in enumerate(page_nums):
        col = nav_cols[i + 1]
        with col:
            if prev_p and p - prev_p > 1:
                _render_html(
                    f'<div style="text-align:center;color:{COLORS["text_dim"]};padding-top:8px;font-size:0.85rem;">…</div>'
                )
            else:
                if p == current_page:
                    _render_html(
                        f'<div style="text-align:center;background:{COLORS["accent"]};color:#0e1117;'
                        f'border-radius:6px;padding:8px 0;font-weight:700;font-size:0.86rem;min-height:38px;'
                        f'display:flex;align-items:center;justify-content:center;">{p}</div>'
                    )
                else:
                    if st.button(str(p), key=f"pl_page_{p}", use_container_width=True):
                        st.session_state["pl_page"] = p
                        st.rerun()
        prev_p = p

    with nav_cols[-1]:
        if st.button("Next →", key="pl_next", disabled=(current_page == total_pages), use_container_width=True):
            st.session_state["pl_page"] = min(total_pages, current_page + 1)
            st.rerun()


# ====================================================================
# Test the matcher — with OOD detection
# ====================================================================
section_header(
    "Test the matcher",
    "See which patterns the AI would surface for a given scenario",
)

_render_html(
    f'<div style="color:{COLORS["text_muted"]};font-size:0.84rem;line-height:1.6;margin-bottom:1rem;">'
    f'Describe a suspicious behavior you\'ve seen and the AI will return the closest matching patterns from the library — '
    f'the same way it does during an investigation. Out-of-domain queries (e.g. cooking questions) are automatically rejected '
    f'using PCA-based subspace detection.'
    f'</div>'
)

rag_query = st.text_input(
    "Describe the scenario",
    placeholder="e.g. small charge then large purchase on same card within an hour",
    label_visibility="collapsed",
    key="rag_query_input",
)

if rag_query:
    try:
        from src.agentic.tools import search_fraud_cases

        with st.spinner("Searching patterns…"):
            results = search_fraud_cases(rag_query, top_k=3)

        # Handle OOD rejection
        if results and isinstance(results[0], dict) and results[0].get("ood_rejected"):
            ood_method = results[0].get("ood_method", "pca")
            ood_reason = results[0].get("reason", "")
            method_label = "PCA subspace check" if ood_method == "pca" else "LLM judge"
            _render_html(
                f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};border-left:3px solid {COLORS["medium"]};'
                f'border-radius:8px;padding:1.2rem 1.4rem;display:flex;gap:14px;align-items:flex-start;margin-top:0.6rem;">'
                f'<div style="flex-shrink:0;margin-top:2px;">{icon("alert-circle", size=18, color=COLORS["medium"])}</div>'
                f'<div>'
                f'<div style="color:{COLORS["text"]};font-size:0.92rem;font-weight:600;margin-bottom:6px;">'
                f'This query doesn\'t look like a fraud question'
                f'</div>'
                f'<div style="color:{COLORS["text_muted"]};font-size:0.84rem;line-height:1.55;margin-bottom:8px;">{ood_reason}</div>'
                f'<div style="color:{COLORS["text_dim"]};font-size:0.74rem;">'
                f'Detected by: <span style="color:{COLORS["accent"]};font-family:JetBrains Mono,monospace;">{method_label}</span>. '
                f'The pattern library only contains fraud-domain patterns. Try a query about suspicious card activity, velocity, geolocation, etc.'
                f'</div>'
                f'</div>'
                f'</div>'
            )
        elif results and isinstance(results[0], dict) and "error" in results[0]:
            st.error(f"Pattern search failed: {results[0]['error']}")
        elif not results:
            empty_state(
                "No matching patterns found",
                "Try describing the behavior differently. Patterns below the quality threshold are filtered out.",
            )
        else:
            _render_html(
                f'<div style="color:{COLORS["text_muted"]};font-size:0.82rem;margin:1rem 0 0.8rem 0;">'
                f'Top {len(results)} matches'
                f'</div>'
            )

            for r in results:
                sim = r.get("similarity")
                sim_str = f"{sim*100:.0f}%" if sim is not None else "—"
                band = r.get("match_band", "")
                pattern_str = r.get("pattern", "?")
                color = PATTERN_COLORS.get(pattern_str, COLORS["text_muted"])
                pattern_label = pattern_str.replace("_", " ").title()

                _render_html(
                    f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};border-left:3px solid {color};'
                    f'padding:0.95rem 1.2rem;border-radius:8px;margin-bottom:0.6rem;">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:8px;">'
                    f'<div style="color:{COLORS["text"]};font-weight:600;font-size:0.92rem;">{r.get("title", "?")}</div>'
                    f'<div style="display:flex;gap:14px;align-items:center;flex-shrink:0;">'
                    f'<span style="background:{color};color:#0e1117;padding:2px 8px;border-radius:4px;font-size:0.66rem;font-weight:700;'
                    f'text-transform:uppercase;letter-spacing:0.06em;">{pattern_label}</span>'
                    f'<div style="display:flex;flex-direction:column;align-items:flex-end;gap:2px;">'
                    f'<span style="color:{COLORS["accent"]};font-size:0.86rem;font-weight:700;font-family:Inter,sans-serif;">{sim_str}</span>'
                    f'<span style="color:{COLORS["text_dim"]};font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.06em;">{band}</span>'
                    f'</div>'
                    f'</div>'
                    f'</div>'
                    f'<div style="color:{COLORS["text_muted"]};font-size:0.84rem;line-height:1.55;">{r.get("snippet", "")[:280]}…</div>'
                    f'</div>'
                )
    except Exception as e:
        st.error(f"Pattern search failed: {e}")
        