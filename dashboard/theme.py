"""
theme.py

Visual system for the dashboard: design tokens, CSS, Plotly styling and small
HTML helpers. Keeping this separate from app.py means the look can change
without touching any analysis code.

Palette (design tokens)
-----------------------
Brand      PRIMARY      #2455d6   actions, selected state, main data colour
Semantic   SUCCESS      #157347   healthy / standard path
           WARNING      #a35b0a   needs attention / skipped steps
           DANGER       #c2410c   the problem: bottleneck, rework, breaches
           INFO         #5b45c9   extended or unusual paths
Neutral    INK / MUTED / LINE / CANVAS / SURFACE for text, borders, backgrounds

Every semantic colour has a SOFT tint used for badge and chip backgrounds.
Colour always carries the same meaning across charts, tables and badges.
"""

from __future__ import annotations

import html

import plotly.graph_objects as go
import streamlit as st

from src.formatting import fmt_minutes  # noqa: F401  (re-exported for the dashboard)

# ── Design tokens ────────────────────────────────────────────────────────────
PRIMARY = "#2455d6"
PRIMARY_DARK = "#1b43ab"
PRIMARY_SOFT = "#e8eefc"
PRIMARY_TINT = "#c9d6f5"

SUCCESS = "#157347"
SUCCESS_SOFT = "#e1f3e8"
WARNING = "#a35b0a"
WARNING_SOFT = "#fdf0d9"
DANGER = "#c2410c"
DANGER_SOFT = "#fbe9e1"
INFO = "#5b45c9"
INFO_SOFT = "#ece9fb"

INK = "#0f172a"
MUTED = "#5b6478"
LINE = "#e3e7ee"
CANVAS = "#f6f7f9"
SURFACE = "#ffffff"
NEUTRAL = "#94a3b8"
NEUTRAL_SOFT = "#eef1f5"

# aliases used by the charts
ACCENT = PRIMARY
ACCENT_SOFT = PRIMARY_TINT
WARN = DANGER
AMBER = WARNING

# Chart fills need more separation than badges do: dark green, dark vermilion and bright
# gold differ in lightness as well as hue, so they stay distinct for colour-blind viewers.
VARIANT_COLORS = {
    "Conformant": SUCCESS,
    "Rework loop": DANGER,
    "Skip": "#e0a100",
    "Extended": INFO,
}
_BADGE_KIND = {"Conformant": "ok", "Rework loop": "risk", "Skip": "warn", "Extended": "info"}

FONT_STACK = "Inter, -apple-system, 'Segoe UI', Roboto, sans-serif"

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {{ font-family: {FONT_STACK}; }}
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1200px; }}
header[data-testid="stHeader"] {{ background: transparent; }}
#MainMenu, footer {{ visibility: hidden; }}

h1, h2, h3 {{ color: {INK}; letter-spacing: -0.01em; }}
h1 {{ font-size: 1.7rem !important; font-weight: 650 !important; margin-bottom: 0.1rem; }}
h2 {{ font-size: 1.15rem !important; font-weight: 600 !important; margin-top: 1.6rem; }}
h3 {{ font-size: 1rem !important; font-weight: 600 !important; }}

/* sidebar */
section[data-testid="stSidebar"] {{ background: {SURFACE}; border-right: 1px solid {LINE}; }}
section[data-testid="stSidebar"] h2 {{ font-size: 0.72rem !important; text-transform: uppercase;
    letter-spacing: 0.08em; color: {MUTED}; font-weight: 600 !important; margin-top: 1.2rem; }}
.brand {{ display: flex; align-items: center; gap: 10px; padding: 4px 0 14px 0;
    border-bottom: 1px solid {LINE}; margin-bottom: 4px; }}
.brand .name {{ font-weight: 650; color: {INK}; font-size: 0.98rem; line-height: 1.15; }}
.brand .sub {{ color: {MUTED}; font-size: 0.74rem; margin-top: 2px; }}
.sidebar-foot {{ color: {MUTED}; font-size: 0.74rem; margin-top: 1.5rem; line-height: 1.4; }}

/* navigation rendered as pills */
.st-key-nav [data-testid="stRadioGroup"] {{ gap: 2px; width: 100%; }}
.st-key-nav label[data-testid="stRadioOption"] {{ display: flex; width: 100%; box-sizing: border-box;
    padding: 6px 12px; margin: 0; border-radius: 8px; transition: background 0.12s; }}
.st-key-nav label[data-testid="stRadioOption"] > div {{ width: 100%; }}
.st-key-nav label[data-testid="stRadioOption"] > div > div:first-child {{ display: none; }}
.st-key-nav label[data-testid="stRadioOption"]:hover {{ background: {NEUTRAL_SOFT}; }}
.st-key-nav label[data-testid="stRadioOption"][data-selected="true"] {{ background: {PRIMARY_SOFT}; }}
.st-key-nav label[data-testid="stRadioOption"][data-selected="true"] * {{ color: {PRIMARY} !important;
    font-weight: 600; }}

.kicker {{ color: {MUTED}; font-size: 0.9rem; margin-bottom: 1.2rem; }}

/* cards */
.finding {{ background: {SURFACE}; border: 1px solid {LINE}; border-left: 4px solid {DANGER};
    border-radius: 8px; padding: 18px 22px; margin: 0.4rem 0 1.2rem 0; }}
.finding .label {{ font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em;
    color: {DANGER}; font-weight: 600; }}
.finding .headline {{ font-size: 1.2rem; font-weight: 600; color: {INK}; margin: 4px 0 6px 0; }}
.finding .detail {{ color: {MUTED}; font-size: 0.95rem; line-height: 1.5; }}

.kpi {{ background: {SURFACE}; border: 1px solid {LINE}; border-radius: 8px; padding: 14px 16px; height: 100%;
    min-height: 98px; }}
.kpi .label {{ font-size: 0.78rem; color: {MUTED}; font-weight: 500; }}
.kpi .value {{ font-size: 1.55rem; font-weight: 650; color: {INK}; line-height: 1.25; margin-top: 2px; }}
.kpi .note {{ font-size: 0.78rem; color: {MUTED}; margin-top: 2px; }}

.chain {{ background: {SURFACE}; border: 1px solid {LINE}; border-radius: 8px; margin-bottom: 1.25rem; }}
.chain .row {{ display: flex; gap: 16px; padding: 14px 20px; border-bottom: 1px solid {LINE}; }}
.chain .row:last-child {{ border-bottom: none; }}
.chain .key {{ flex: 0 0 90px; font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.07em;
    color: {MUTED}; font-weight: 600; padding-top: 3px; }}
.chain .val {{ color: {INK}; line-height: 1.5; }}

/* badges */
.badge {{ display: inline-block; padding: 2px 9px; border-radius: 999px; font-size: 0.74rem; font-weight: 600;
    line-height: 1.5; white-space: nowrap; }}
.badge.ok {{ background: {SUCCESS_SOFT}; color: {SUCCESS}; }}
.badge.warn {{ background: {WARNING_SOFT}; color: {WARNING}; }}
.badge.risk {{ background: {DANGER_SOFT}; color: {DANGER}; }}
.badge.info {{ background: {INFO_SOFT}; color: {INFO}; }}
.badge.muted {{ background: {NEUTRAL_SOFT}; color: {MUTED}; }}

/* variants table */
.vt {{ width: 100%; border-collapse: separate; border-spacing: 0; background: {SURFACE};
    border: 1px solid {LINE}; border-radius: 8px; overflow: hidden; font-size: 0.9rem; }}
.vt th {{ text-align: left; font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.06em;
    color: {MUTED}; font-weight: 600; padding: 10px 14px; background: #fafbfc; border-bottom: 1px solid {LINE}; }}
.vt td {{ padding: 11px 14px; border-bottom: 1px solid {LINE}; color: {INK}; vertical-align: middle; }}
.vt tr:last-child td {{ border-bottom: none; }}
.vt .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
.vt .bar {{ display: inline-block; width: 70px; height: 6px; background: {NEUTRAL_SOFT}; border-radius: 3px;
    vertical-align: middle; margin-right: 8px; overflow: hidden; }}
.vt .bar span {{ display: block; height: 100%; border-radius: 3px; }}
.vt .bar.wide {{ width: 150px; }}
.vt td.muted {{ color: {MUTED}; }}

/* variant path cards */
.vcard {{ background: {SURFACE}; border: 1px solid {LINE}; border-radius: 8px; padding: 14px 18px; margin-bottom: 10px; }}
.vcard .head {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }}
.vcard .title {{ font-weight: 600; color: {INK}; }}
.vcard .meta {{ color: {MUTED}; font-size: 0.85rem; }}
.vcard .action {{ color: {MUTED}; font-size: 0.88rem; line-height: 1.5; margin-top: 10px; }}
.path {{ display: flex; flex-wrap: wrap; align-items: center; gap: 6px; margin-top: 10px; }}
.chip {{ padding: 3px 10px; border-radius: 6px; font-size: 0.8rem; background: {NEUTRAL_SOFT}; color: {INK};
    border: 1px solid {LINE}; }}
.chip.repeat {{ background: {DANGER_SOFT}; border-color: #f2c5b0; color: {DANGER}; font-weight: 600; }}
.chip.extra {{ background: {INFO_SOFT}; border-color: #d4cdf3; color: {INFO}; font-weight: 600; }}
.chip.missing {{ background: transparent; border: 1px dashed {WARNING}; color: {WARNING}; font-weight: 600; }}
.arrow {{ color: {NEUTRAL}; font-size: 0.8rem; }}

.note {{ color: {MUTED}; font-size: 0.85rem; }}
div[data-testid="stDataFrame"] {{ border: 1px solid {LINE}; border-radius: 8px; overflow: hidden; }}
div[data-testid="stAlert"] {{ border-radius: 8px; }}
</style>
"""

_LOGO = (
    '<svg width="30" height="30" viewBox="0 0 30 30" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
    f'<rect width="30" height="30" rx="7" fill="{PRIMARY}"/>'
    '<rect x="6" y="7" width="18" height="3.4" rx="1.7" fill="#fff"/>'
    '<rect x="10.5" y="13.3" width="9" height="3.4" rx="1.7" fill="#ffb48a"/>'
    '<rect x="6" y="19.6" width="18" height="3.4" rx="1.7" fill="#fff"/></svg>'
)


def apply_theme() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def esc(value) -> str:
    """Escape any text that came from the uploaded file before putting it in HTML."""
    return html.escape(str(value))


def brand() -> str:
    return (f'<div class="brand">{_LOGO}<div><div class="name">Bottleneck Analyzer</div>'
            '<div class="sub">Workflow analytics</div></div></div>')


def badge(text: str, kind: str = "muted") -> str:
    return f'<span class="badge {kind}">{esc(text)}</span>'


def type_badge(variant_type: str) -> str:
    return badge(variant_type, _BADGE_KIND.get(variant_type, "muted"))


def status_kind(status: str) -> str:
    return {"Healthy": "ok", "Needs attention": "warn", "At risk": "risk"}.get(status, "muted")


def kpi(label: str, value: str, note: str = "", status: str | None = None) -> str:
    if status:
        note_html = f'<div class="note">{badge(status, status_kind(status))}</div>'
    elif note:
        note_html = f'<div class="note">{esc(note)}</div>'
    else:
        note_html = ""
    return (f'<div class="kpi"><div class="label">{esc(label)}</div>'
            f'<div class="value">{esc(value)}</div>{note_html}</div>')


def finding_card(label: str, headline: str, detail: str) -> str:
    return (f'<div class="finding"><div class="label">{esc(label)}</div>'
            f'<div class="headline">{esc(headline)}</div>'
            f'<div class="detail">{esc(detail)}</div></div>')


def chain_card(rows: list[tuple[str, str]]) -> str:
    body = "".join(
        f'<div class="row"><div class="key">{esc(k)}</div><div class="val">{esc(v)}</div></div>'
        for k, v in rows
    )
    return f'<div class="chain">{body}</div>'


def _bar(value: float) -> str:
    color = DANGER if value >= 50 else WARNING if value >= 25 else PRIMARY
    return (f'<span class="bar"><span style="width:{min(value, 100):.0f}%;background:{color}"></span></span>'
            f'{value:.0f}%')


def risk_bar(value: float) -> str:
    """Bar plus number for a 0-100 index."""
    color = DANGER if value >= 70 else WARNING if value >= 40 else PRIMARY
    return (f'<span class="bar wide"><span style="width:{min(max(value, 0), 100):.0f}%;'
            f'background:{color}"></span></span>{value:.0f}')


def data_table(headers: list[str], rows: list[list[str]], numeric: set[int] | None = None) -> str:
    """Table whose text wraps instead of scrolling sideways.

    ``rows`` hold ready-made HTML cells: escape text with ``esc`` first. Columns
    listed in ``numeric`` are right-aligned.
    """
    numeric = numeric or set()
    head = "".join(
        f"<th class='{'num' if i in numeric else ''}'>{esc(h)}</th>" for i, h in enumerate(headers)
    )
    body = "".join(
        "<tr>" + "".join(
            f"<td class='{'num' if i in numeric else ''}'>{cell}</td>" for i, cell in enumerate(row)
        ) + "</tr>"
        for row in rows
    )
    return f"<table class='vt'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def variants_table(rows: list[dict]) -> str:
    """Compact table that fits the page width (no horizontal scrolling)."""
    body = "".join(
        "<tr>"
        f"<td><b>{esc(r['variant'])}</b></td>"
        f"<td>{type_badge(r['type'])}</td>"
        f"<td class='num'>{r['cases']:,}</td>"
        f"<td class='num'>{r['share']:.0f}%</td>"
        f"<td class='num'>{r['avg_cycle']:.1f} h</td>"
        f"<td>{_bar(r['breach'])}</td>"
        f"<td class='num'>{r['wait']:.0f}%</td>"
        "</tr>"
        for r in rows
    )
    head = ("<tr><th>Variant</th><th>Type</th><th class='num'>Cases</th><th class='num'>Share</th>"
            "<th class='num'>Avg cycle</th><th>SLA breaches</th><th class='num'>Waiting share</th></tr>")
    return f"<table class='vt'><thead>{head}</thead><tbody>{body}</tbody></table>"


def path_chips(path: list[str], standard: list[str]) -> str:
    """Show a variant's steps as chips; repeated steps and skipped steps stand out."""
    seen: set[str] = set()
    chips = []
    for step in path:
        kind = "repeat" if step in seen else ("extra" if step not in standard else "")
        seen.add(step)
        chips.append(f'<span class="chip {kind}">{esc(step)}</span>')
    inner = '<span class="arrow">&rsaquo;</span>'.join(chips)
    missing = [s for s in standard if s not in path]
    for step in missing:
        inner += f'<span class="chip missing">skipped: {esc(step)}</span>'
    return f'<div class="path">{inner}</div>'


def variant_card(label: str, variant_type: str, meta: str, path: list[str],
                 standard: list[str], action: str) -> str:
    return (f'<div class="vcard"><div class="head"><span class="title">{esc(label)}</span>'
            f'{type_badge(variant_type)}<span class="meta">{esc(meta)}</span></div>'
            f'{path_chips(path, standard)}<div class="action">{esc(action)}</div></div>')


def style_fig(fig: go.Figure, height: int = 340, legend: bool = True) -> go.Figure:
    """Apply the shared chart look: quiet grid, no chart title, readable type."""
    fig.update_layout(
        height=height,
        font=dict(family=FONT_STACK, size=12, color=INK),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=None),
        hoverlabel=dict(font_family=FONT_STACK),
    )
    fig.update_xaxes(showgrid=False, linecolor=LINE, ticks="outside", tickcolor=LINE, title_font_size=12,
                     automargin=True)
    fig.update_yaxes(gridcolor=LINE, zeroline=False, title_font_size=12, automargin=True)
    return fig


CHART_CONFIG = {"displayModeBar": False}