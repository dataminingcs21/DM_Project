"""Visual layer of the application: colours, CSS, header, cards and badges.

Kept apart from ``ui_components.py`` (logic + standard widgets) so the look can change without touching
what the page does.  Colours match ``.streamlit/config.toml``.
"""

from __future__ import annotations

import html

import streamlit as st

ACCENT = "#22d3ee"
BENIGN = "#10b981"
ATTACK = "#f43f5e"
WARN = "#f59e0b"
MUTED = "#8b9bb4"
CARD = "#131c2e"
BORDER = "#22304a"

CSS = f"""
<style>
.block-container {{ padding-top: 2.2rem; max-width: 1280px; }}
h1 {{ font-weight: 750 !important; letter-spacing: -0.02em; }}
.nids-hero {{
  background: linear-gradient(120deg, rgba(34,211,238,0.14), rgba(16,185,129,0.06) 55%, rgba(244,63,94,0.06));
  border: 1px solid {BORDER}; border-radius: 16px; padding: 18px 22px 14px; margin: -0.6rem 0 1.1rem;
}}
.nids-hero p {{ color: {MUTED}; margin: 0 0 10px; font-size: 0.98rem; }}
.nids-pill {{
  display: inline-block; padding: 3px 11px; margin: 0 6px 4px 0; border-radius: 999px; font-size: 0.78rem;
  border: 1px solid {BORDER}; background: rgba(255,255,255,0.03); color: #c9d6ea;
}}
.nids-kpi {{
  background: {CARD}; border: 1px solid {BORDER}; border-radius: 14px; padding: 14px 16px; height: 100%;
}}
.nids-kpi .label {{ color: {MUTED}; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; }}
.nids-kpi .value {{ font-size: 1.65rem; font-weight: 700; margin-top: 2px; }}
.nids-kpi .sub {{ color: {MUTED}; font-size: 0.8rem; }}
.nids-verdict {{
  border-radius: 14px; padding: 16px 20px; margin: 4px 0 14px; border: 1px solid;
  display: flex; gap: 14px; align-items: center;
}}
.nids-verdict .icon {{ font-size: 1.9rem; line-height: 1; }}
.nids-verdict .title {{ font-size: 1.15rem; font-weight: 700; }}
.nids-verdict .text {{ color: #c9d6ea; font-size: 0.92rem; }}
.nids-verdict.ok {{ background: rgba(16,185,129,0.10); border-color: rgba(16,185,129,0.45); }}
.nids-verdict.bad {{ background: rgba(244,63,94,0.10); border-color: rgba(244,63,94,0.50); }}
.nids-steps {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin: 6px 0 14px; }}
.nids-step {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 12px; padding: 10px 14px; }}
.nids-step b {{ color: {ACCENT}; margin-right: 6px; }}
.nids-step span {{ color: #c9d6ea; font-size: 0.88rem; }}
.nids-note {{ color: {MUTED}; font-size: 0.82rem; }}
@media (max-width: 760px) {{ .nids-steps {{ grid-template-columns: 1fr; }} }}
div[data-testid="stTabs"] button p {{ font-size: 0.98rem; font-weight: 600; }}
div[data-testid="stFileUploader"] section {{ border: 1.5px dashed {BORDER}; border-radius: 14px; }}
</style>
"""


def inject_css() -> None:
    st.html(CSS)


def hero(subtitle: str, pills: list[str]) -> None:
    pills_html = "".join(f'<span class="nids-pill">{html.escape(p)}</span>' for p in pills)
    st.html(f'<div class="nids-hero"><p>{html.escape(subtitle)}</p>{pills_html}</div>')


def kpi(label: str, value: str, sub: str = "", color: str | None = None) -> None:
    style = f' style="color:{color}"' if color else ""
    st.html(
        f'<div class="nids-kpi"><div class="label">{html.escape(label)}</div>'
        f'<div class="value"{style}>{html.escape(value)}</div><div class="sub">{html.escape(sub)}</div></div>'
    )


def kpi_row(items: list[tuple[str, str, str, str | None]]) -> None:
    for column, (label, value, sub, color) in zip(st.columns(len(items)), items):
        with column:
            kpi(label, value, sub, color)


def verdict(ok: bool, title: str, text: str) -> None:
    icon = "&#x1F6E1;" if ok else "&#x26A0;"
    st.html(
        f'<div class="nids-verdict {"ok" if ok else "bad"}"><div class="icon">{icon}</div>'
        f'<div><div class="title">{html.escape(title)}</div><div class="text">{html.escape(text)}</div></div></div>'
    )


def steps(items: list[tuple[str, str]]) -> None:
    cells = "".join(f'<div class="nids-step"><b>{html.escape(n)}</b><span>{html.escape(t)}</span></div>' for n, t in items)
    st.html(f'<div class="nids-steps">{cells}</div>')


def note(text: str) -> None:
    st.html(f'<div class="nids-note">{html.escape(text)}</div>')
