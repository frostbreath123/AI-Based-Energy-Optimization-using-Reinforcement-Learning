"""Plotly figure builders. All colours come from app.style tokens."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.style import C, JOB_COLORS, POLICY_COLORS

FONT = dict(family="Inter, Segoe UI, sans-serif", size=12, color=C["ink"])


def _base(fig: go.Figure, height: int = 320, title: str | None = None, legend: bool = True) -> go.Figure:
    layout = dict(height=height + (46 if legend else 0), margin=dict(l=8, r=8, t=40 if title else 12, b=8), font=FONT,
                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", hovermode="x unified",
                  showlegend=legend, legend=dict(orientation="h", yanchor="top", y=-0.32, x=0, bgcolor="rgba(0,0,0,0)"))
    if title:
        layout["title"] = dict(text=title, x=0, font=dict(size=14))
    fig.update_layout(**layout)
    fig.update_xaxes(showgrid=False, linecolor=C["line"], ticks="outside", tickcolor=C["line"])
    fig.update_yaxes(gridcolor="#ECEEF2", zeroline=False)
    return fig


def hours_axis(fig: go.Figure) -> go.Figure:
    fig.update_xaxes(tickmode="linear", tick0=0, dtick=3, range=[-0.2, 24.2], title_text="Hour of day")
    return fig


def fig_signals(sc, H: int = 24) -> go.Figure:
    x = np.arange(H)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=x, y=sc.price[:H], mode="lines", line=dict(color=C["price"], width=2.4, shape="hv"),
                             name="Tariff (EUR/kWh)", hovertemplate="%{y:.3f} EUR/kWh"), secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=sc.carbon[:H], mode="lines", line=dict(color=C["carbon"], width=1.8, dash="dot", shape="hv"),
                             name="Grid carbon (g/kWh)", hovertemplate="%{y:.0f} gCO2e/kWh"), secondary_y=True)
    fig.update_yaxes(title_text="EUR/kWh", secondary_y=False)
    fig.update_yaxes(title_text="gCO2e/kWh", secondary_y=True, showgrid=False)
    return hours_axis(_base(fig, 280, "Electricity tariff and grid carbon intensity"))


def fig_renewables(sc, H: int = 24) -> go.Figure:
    x = np.arange(H)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=sc.pv[:H], name="PV actual", mode="lines", fill="tozeroy",
                             line=dict(color=C["pv"], width=2, shape="hv"), fillcolor="rgba(217,164,4,.22)",
                             hovertemplate="%{y:.0f} kW"))
    fig.add_trace(go.Scatter(x=x, y=sc.pv_forecast[:H], name="PV forecast (day-ahead)", mode="lines",
                             line=dict(color=C["pv"], width=1.6, dash="dash", shape="hv"), hovertemplate="%{y:.0f} kW"))
    fig.add_trace(go.Scatter(x=x, y=sc.base_kw[:H], name="Inflexible base load", mode="lines",
                             line=dict(color=C["muted"], width=1.8, shape="hv"), hovertemplate="%{y:.0f} kW"))
    fig.update_yaxes(title_text="kW")
    return hours_axis(_base(fig, 280, "Renewable availability vs inflexible demand"))


def fig_dispatch(trace: list[dict], n_jobs: int, title: str, H: int = 24) -> go.Figure:
    t = [r["t"] for r in trace]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=t, y=[r["base"] for r in trace], name="Inflexible base load", marker_color=C["base"],
                         hovertemplate="%{y:.0f} kW"))
    for k in range(n_jobs):
        fig.add_trace(go.Bar(x=t, y=[r["jobs"][k] for r in trace], name=f"Job {k + 1}",
                             marker_color=JOB_COLORS[k % len(JOB_COLORS)], hovertemplate="%{y:.0f} kW"))
    fig.add_trace(go.Bar(x=t, y=[max(r["battery_kw"], 0) for r in trace], name="Battery charging",
                         marker_color=C["battery"], hovertemplate="%{y:.0f} kW"))
    fig.add_trace(go.Bar(x=t, y=[min(r["battery_kw"], 0) for r in trace], name="Battery discharging",
                         marker_color="#B7A5E6", hovertemplate="%{y:.0f} kW"))
    fig.add_trace(go.Scatter(x=t, y=[r["pv"] for r in trace], name="PV generation", mode="lines",
                             line=dict(color=C["pv"], width=2.4, shape="hv"), hovertemplate="%{y:.0f} kW"))
    fig.add_trace(go.Scatter(x=t, y=[r["import"] for r in trace], name="Grid import", mode="lines",
                             line=dict(color=C["grid"], width=2, dash="dot", shape="hv"), hovertemplate="%{y:.0f} kW"))
    fig.update_layout(barmode="relative", bargap=0.08)
    fig.update_yaxes(title_text="kW")
    return hours_axis(_base(fig, 360, title))


def fig_gantt(sim, title: str) -> go.Figure:
    fig = go.Figure()
    n = sim.n_jobs
    ylab = [f"Job {k + 1} ({sim.jobs[k].power_kw:.0f} kW)" for k in range(n)]
    for k, j in enumerate(sim.jobs):
        fig.add_trace(go.Bar(y=[ylab[k]], x=[j.duration], base=[j.planned_start], orientation="h",
                             marker=dict(color="rgba(0,0,0,0)", line=dict(color=C["naive"], width=1.5)),
                             name="Planned window", showlegend=(k == 0), hovertemplate="planned %{base:.0f}-%{x:.0f} h<extra></extra>"))
        hrs = [r["t"] for r in sim.trace if r["jobs"][k] > 0]
        fig.add_trace(go.Bar(y=[ylab[k]] * len(hrs), x=[1] * len(hrs), base=hrs, orientation="h",
                             marker_color=JOB_COLORS[k % len(JOB_COLORS)], name="Executed", showlegend=(k == 0),
                             hovertemplate="runs hour %{base:.0f}<extra></extra>"))
        fig.add_trace(go.Scatter(y=[ylab[k]], x=[j.deadline], mode="markers", name="Deadline", showlegend=(k == 0),
                                 marker=dict(symbol="diamond", size=11, color=C["bad"]),
                                 hovertemplate="deadline hour %{x}<extra></extra>"))
        fig.add_trace(go.Scatter(y=[ylab[k]], x=[j.release], mode="markers", name="Released", showlegend=(k == 0),
                                 marker=dict(symbol="triangle-right", size=9, color=C["muted"]),
                                 hovertemplate="released hour %{x}<extra></extra>"))
    fig.update_layout(barmode="overlay", hovermode="closest")
    fig.update_yaxes(autorange="reversed", gridcolor="#ECEEF2")
    return hours_axis(_base(fig, 60 + 38 * max(n, 1), title))


def fig_soc(traces: dict[str, list[dict]], cap: float) -> go.Figure:
    fig = go.Figure()
    for name, tr in traces.items():
        fig.add_trace(go.Scatter(x=[r["t"] + 1 for r in tr], y=[100 * r["soc"] / cap for r in tr], name=name, mode="lines",
                                 line=dict(color=POLICY_COLORS[name], width=2.2), hovertemplate="%{y:.0f} %"))
    fig.update_yaxes(title_text="State of charge (%)", range=[0, 100])
    return hours_axis(_base(fig, 240, "Battery state of charge"))


def fig_policy_bars(metrics: dict[str, dict], keys: list[tuple[str, str, str]]) -> go.Figure:
    """keys: (metric, title, unit). One small bar panel per metric."""
    fig = make_subplots(rows=1, cols=len(keys), subplot_titles=[f"{t} ({u})" for _, t, u in keys])
    for i, (m, _, _) in enumerate(keys, start=1):
        for pol, vals in metrics.items():
            fig.add_trace(go.Bar(x=[pol], y=[vals[m]], marker_color=POLICY_COLORS[pol], name=pol, showlegend=(i == 1),
                                 text=[f"{vals[m]:,.0f}"], textposition="outside", hovertemplate="%{y:,.1f}"), row=1, col=i)
    fig.update_layout(bargap=0.35, hovermode="closest")
    fig.update_xaxes(showticklabels=False)
    for i, (m, _, _) in enumerate(keys, start=1):
        top = max(v[m] for v in metrics.values())
        fig.update_yaxes(range=[0, top * 1.2 if top > 0 else 1], row=1, col=i)
    out = _base(fig, 300)
    out.update_layout(margin=dict(l=8, r=8, t=40, b=8))
    return out


def fig_heatmap(z: np.ndarray, x: list, y: list, title: str, zlabel: str, mid: float | None = 0.0) -> go.Figure:
    fig = go.Figure(go.Heatmap(z=z, x=x, y=y, colorscale=[[0, "#C2410C"], [0.5, "#F7F7F5"], [1, "#0E7C6B"]],
                               zmid=mid, colorbar=dict(title=zlabel, thickness=12),
                               text=np.round(z, 1), texttemplate="%{text}", hovertemplate="%{y} | %{x}: %{z:.1f}<extra></extra>"))
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(autorange="reversed")
    return _base(fig, 60 + 34 * len(y), title, legend=False)


def fig_box(df: pd.DataFrame, x: str, y: str, color: str, title: str, ytitle: str) -> go.Figure:
    fig = go.Figure()
    for pol, g in df.groupby(color, sort=False):
        fig.add_trace(go.Box(x=g[x], y=g[y], name=pol, marker_color=POLICY_COLORS.get(pol, C["muted"]),
                             boxpoints="outliers", line=dict(width=1.3)))
    fig.update_layout(boxmode="group", hovermode="closest")
    fig.update_yaxes(title_text=ytitle)
    return _base(fig, 360, title)


def fig_scatter(df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    for pol, g in df.groupby("group", sort=False):
        fig.add_trace(go.Scatter(x=g.cost_eur, y=g.co2_kg, mode="markers", name=pol,
                                 marker=dict(size=7, color=POLICY_COLORS.get(pol, C["muted"]), opacity=0.75,
                                             line=dict(width=0.5, color="white")),
                                 hovertemplate="cost %{x:,.0f} EUR | CO2 %{y:,.0f} kg<extra></extra>"))
    fig.update_xaxes(title_text="Daily electricity cost (EUR)", showgrid=True)
    fig.update_yaxes(title_text="Daily emissions (kg CO2e)")
    fig.update_layout(hovermode="closest")
    return _base(fig, 360, title)
