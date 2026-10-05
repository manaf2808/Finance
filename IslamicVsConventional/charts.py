"""Plotly figures."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import STRESS_WINDOW, TRADING_DAYS

ISL, CONV, STRESS = "#1fc416", "#122bcd", "#c0392b"


def rolling_vol(r: pd.Series) -> pd.Series:
    return r.rolling(STRESS_WINDOW).std() * np.sqrt(TRADING_DAYS) * 100


def shade(fig, index, episodes):
    for s, e in episodes:
        fig.add_vrect(x0=index[s], x1=index[e], fillcolor="red", opacity=0.12, line_width=0)


def daily_returns_chart(combined, isl, conv):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=combined.index, y=combined["conventional"] * 100, name=conv,
                             line=dict(color=CONV, width=0.8)))
    fig.add_trace(go.Scatter(x=combined.index, y=combined["islamic"] * 100, name=isl,
                             line=dict(color=ISL, width=0.8)))
    fig.update_layout(title="Daily returns (%)", yaxis_title="%", height=350, legend=dict(orientation="h"))
    return fig


def volatility_chart(combined, isl, conv):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=combined.index, y=rolling_vol(combined["islamic"]), name=isl, line=dict(color=ISL)))
    fig.add_trace(go.Scatter(x=combined.index, y=rolling_vol(combined["conventional"]), name=conv,
                             line=dict(color=CONV)))
    fig.update_layout(title=f"Annualised volatility ({STRESS_WINDOW}-day rolling, %)", yaxis_title="%",
                      height=350, legend=dict(orientation="h"))
    return fig


def stress_vs_funds_chart(stress, combined, isl, conv, label, threshold, episodes, own_axis=False):
    """Stress measure vs the two funds' volatility, crisis periods shaded."""
    fig = make_subplots(specs=[[{"secondary_y": own_axis}]])
    fig.add_trace(go.Scatter(x=stress.index, y=stress, name=label, line=dict(color=STRESS, width=1.4)),
                  secondary_y=False if not own_axis else True)
    fig.add_trace(go.Scatter(x=combined.index, y=rolling_vol(combined["conventional"]), name=f"{conv} volatility",
                             line=dict(color=CONV, width=1)), secondary_y=False)
    fig.add_trace(go.Scatter(x=combined.index, y=rolling_vol(combined["islamic"]), name=f"{isl} volatility",
                             line=dict(color=ISL, width=1)), secondary_y=False)
    fig.add_hline(y=threshold, line_dash="dash", line_color=STRESS, secondary_y=own_axis if own_axis else None,
                  annotation_text="crisis threshold", annotation_position="top left")
    shade(fig, combined.index, episodes)
    fig.update_yaxes(title_text="Annualised volatility (%)", secondary_y=False)
    if own_axis:
        fig.update_yaxes(title_text=label, secondary_y=True, showgrid=False)
    fig.update_layout(height=420, legend=dict(orientation="h"))
    return fig


def _threshold_line(fig, threshold):
    if threshold is not None:
        fig.add_hline(y=threshold, line_dash="dash", line_color=STRESS, annotation_text="crisis threshold",
                      annotation_position="top left")


def components_chart(parts: pd.DataFrame, avg: pd.Series, threshold, title: str):
    """Each firm's volatility (thin) and the average (thick), with the crisis threshold."""
    palette = ["#122bcd", "#f58518", "#1fc416", "#e45756", "#72b7b2"]
    fig = go.Figure()
    for c, col in zip(parts.columns, palette):
        fig.add_trace(go.Scatter(x=parts.index, y=parts[c], name=c, line=dict(color=col, width=0.8), opacity=0.6))
    fig.add_trace(go.Scatter(x=avg.index, y=avg, name="Average (the index)", line=dict(color="#18f1f5", width=2)))
    fig.add_hline(y=avg.mean(), line_dash="dot", line_color="grey", annotation_text="average",
                  annotation_position="bottom left")
    _threshold_line(fig, threshold)
    fig.update_layout(title=title, yaxis_title="Annualised volatility (%)", height=400, legend=dict(orientation="h"))
    return fig


def gpr_chart(g: pd.DataFrame, threshold):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g.index, y=g["GPRD"], name="Daily GPR", line=dict(color="#9ecae1", width=0.7)))
    fig.add_trace(go.Scatter(x=g.index, y=g["GPRD_MA7"], name="7-day average (the index)",
                             line=dict(color="#18f1f5", width=1.8)))
    fig.add_hline(y=g["GPRD_MA7"].mean(), line_dash="dot", line_color="grey", annotation_text="average",
                  annotation_position="bottom left")
    _threshold_line(fig, threshold)
    fig.update_layout(title="Geopolitical Risk index: daily value and 7-day average", yaxis_title="Index (100 = 1985–2019 average)",
                      height=400, legend=dict(orientation="h"))
    return fig


def single_index_chart(s: pd.Series, threshold, title: str, ytitle: str, zero_line=False):
    fig = go.Figure(go.Scatter(x=s.index, y=s, name=title, line=dict(color="#18f1f5", width=1.3)))
    fig.add_hline(y=s.mean(), line_dash="dot", line_color="grey", annotation_text="average",
                  annotation_position="bottom left")
    if zero_line:
        fig.add_hline(y=0, line_color="#18f1f5", line_width=1)
    _threshold_line(fig, threshold)
    fig.update_layout(title=title, yaxis_title=ytitle, height=380, showlegend=False)
    return fig

