"""
EVA Guardian — UI Styles
Minimal, clean styling that enhances Streamlit's native dark theme
without overriding its natural feel.
"""


def get_premium_css() -> str:
    """Returns minimal CSS that complements Streamlit's built-in dark theme."""
    return """
    <style>
    /* Reduce top padding */
    div.block-container { padding-top: 2rem !important; }

    /* Subtle header bar */
    .eva-header {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.2rem 1.8rem;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .eva-header .title {
        font-size: 1.6rem;
        font-weight: 700;
    }
    .eva-header .subtitle {
        color: #9ca3af;
        font-size: 0.85rem;
        margin-top: 0.15rem;
    }
    .eva-header .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(0, 200, 83, 0.1);
        border: 1px solid rgba(0, 200, 83, 0.25);
        color: #4caf50;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 50px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .eva-header .status-dot {
        width: 6px;
        height: 6px;
        background: #4caf50;
        border-radius: 50%;
    }
    </style>
    """


def render_header() -> str:
    """Returns the HTML for the header bar."""
    return """
    <div class="eva-header">
        <div>
            <div class="title">EVA Guardian</div>
            <div class="subtitle">AI-Powered Safety Monitoring for Space Stations</div>
        </div>
        <div class="status-badge"><div class="status-dot"></div>System Online</div>
    </div>
    """


def render_history_counter(counts: dict) -> str:
    """Returns a simple HTML counter for the sidebar."""
    rows = "".join(
        f"<div style='display:flex;justify-content:space-between;padding:2px 0;'>"
        f"<span style='color:#9ca3af;font-size:0.85rem;'>{label}</span>"
        f"<span style='font-weight:700;font-size:0.9rem;'>{val}</span></div>"
        for label, val in counts.items()
    )
    return (
        f"<div style='margin-top:1rem;padding:0.75rem;border:1px solid rgba(255,255,255,0.08);"
        f"border-radius:8px;'>"
        f"<div style='font-size:0.72rem;font-weight:600;text-transform:uppercase;"
        f"letter-spacing:1px;color:#6b7280;margin-bottom:0.5rem;'>Session Detections</div>"
        f"{rows}</div>"
    )
