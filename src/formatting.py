"""Small display helpers shared by the analysis code and the dashboard."""

from __future__ import annotations


def fmt_minutes(minutes: float) -> str:
    """Human-readable duration: 45 min, 5.3 h, 2.1 d."""
    if minutes < 90:
        return f"{minutes:,.0f} min"
    if minutes < 48 * 60:
        return f"{minutes / 60:,.1f} h"
    return f"{minutes / 1440:,.1f} d"