"""
Chart Theme Module for AirQuality System
Enforces unified Green & Yellow environmental styling across all Plotly figures.
Ensures high contrast, legible fonts, subtle gridlines, and dark headings.
"""

import plotly.graph_objects as go


def apply_chart_theme(fig, title=None, x_title=None, y_title=None):
    """
    Applies the standardized Environmental Green & Yellow theme to any Plotly figure:
    - Chart title: Dark Forest Green (#14532D), bold, 15-16px
    - Axis labels: Slate Dark Gray (#374151), 13px
    - Axis tick marks: Slate Gray (#374151), 11px
    - Legend text: Slate Dark Gray (#374151), 12px
    - Background: Clean White (#FFFFFF)
    - Gridlines: Subtle Gray (#F3F4F6)
    - Zero lines & borders: (#E5E7EB)
    """
    if fig is None:
        return fig

    # Determine title
    existing_title = fig.layout.title.text if (fig.layout.title and hasattr(fig.layout.title, "text")) else ""
    final_title = title if title is not None else existing_title

    layout_updates = {
        "paper_bgcolor": "#FFFFFF",
        "plot_bgcolor": "#FFFFFF",
        "font": dict(family="Inter, -apple-system, BlinkMacSystemFont, sans-serif", color="#374151", size=12),
        "legend": dict(
            font=dict(family="Inter, sans-serif", color="#374151", size=12),
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="#E5E7EB",
            borderwidth=1,
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        "margin": dict(l=50, r=40, t=60, b=50)
    }

    if final_title:
        layout_updates["title"] = dict(
            text=f"<b>{final_title}</b>",
            font=dict(family="Inter, sans-serif", size=15, color="#14532D"),
            x=0.01,
            xanchor="left"
        )

    fig.update_layout(**layout_updates)

    # Style axes
    fig.update_xaxes(
        title_font=dict(color="#374151", size=13),
        tickfont=dict(color="#374151", size=11),
        gridcolor="#F3F4F6",
        zerolinecolor="#E5E7EB",
        linecolor="#D1D5DB"
    )

    fig.update_yaxes(
        title_font=dict(color="#374151", size=13),
        tickfont=dict(color="#374151", size=11),
        gridcolor="#F3F4F6",
        zerolinecolor="#E5E7EB",
        linecolor="#D1D5DB"
    )

    if x_title:
        fig.update_xaxes(title_text=x_title)
    if y_title:
        fig.update_yaxes(title_text=y_title)

    # Update subplot titles if present
    for annotation in fig["layout"]["annotations"]:
        annotation["font"] = dict(
            family="Inter, sans-serif",
            size=14,
            color="#14532D"
        )

    return fig

