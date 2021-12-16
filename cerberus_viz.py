from collections import defaultdict
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLORS = [0.1, 0.9]
COLOR_NAMES = ["Red", "Blue"]
COLOR_RGBS = ["rgb(178,24,43)", "rgb(5,48,97)"]

LIGHT = "lightslategrey"
DARK = "#596A7B"

RESOURCE_COLORS = {
    "wood": "springgreen",
    "coal": "darkgray",
    "uranium": "cyan",
}


def is_night(turn):
    return (turn % 40) >= 30


def get_heatmap_trace(state, heatmap_function, opacity=0.9):
    if heatmap_function:
        try:
            array = heatmap_function(state)
        except Exception as e:
            array = None
            print(heatmap_function)
            print(e)
    if isinstance(array, np.ndarray):
        return go.Heatmap(
            name="heatmap",
            z=array.T,
            opacity=opacity,
            colorscale="RdBu",
            showscale=False,
            zmin=-0.2,
            zmax=0.2,
        )


def get_timeseries_traces(state, timeseries_function):
    if timeseries_function:
        try:
            data = timeseries_function(state)
        except Exception as e:
            data = None
            print(timeseries_function)
            print(e)
    traces = list()
    if data is not None:
        rgb = COLOR_RGBS[state.id]
        for name, ydata in data.items():
            traces.append(
                go.Scatter(
                    name=name,
                    x=list(range(len(ydata))),
                    y=ydata,
                    mode="lines",
                    marker=dict(color=rgb),
                )
            )
            traces.append(
                go.Scatter(
                    name=name + "_last",
                    x=[state.turn],
                    y=ydata[-1:],
                    mode="markers",
                    marker=dict(color=rgb),
                )
            )
    return traces


def get_tooltip(data):
    size = len(list(data.values())[0])
    text = ["" for _ in range(size)]
    for key, values in data.items():
        if key in ("x", "y", "team"):
            continue
        for i, v in enumerate(values):
            if v is not None:
                if isinstance(v, (float, np.float32)):
                    text[i] += f"{key}: {v:.3f}<br>"
                else:
                    text[i] += f"{key}: {v}<br>"
    return text


def get_traces(state, cityhighlight_function):

    traces = list()
    colors = COLORS
    resource_colors = RESOURCE_COLORS

    all_cities = deepcopy(state.players[0].cities)
    all_cities.update(state.players[1].cities)

    all_units = deepcopy(state.players[0].units)
    all_units.extend(state.players[1].units)

    unit = defaultdict(list)
    unit_count_at_pos = defaultdict(int)
    for u in all_units:
        unit_count_at_pos[(u.pos.x, u.pos.y)] += 1
        unit["id"].append(u.id)
        unit["x"].append(u.pos.x)
        unit["y"].append(u.pos.y)
        unit["team"].append(u.team)
        unit["cooldown"].append(u.cooldown)
        unit["cargo"].append(u.cargo)

    highlight = defaultdict(list)
    city = defaultdict(list)
    resource = defaultdict(list)
    unit_count = defaultdict(list)
    for i in range(state.map_width):
        for j in range(state.map_height):
            cell = state.map.get_cell(i, j)
            ucount = unit_count_at_pos[(cell.pos.x, cell.pos.y)]
            if ucount > 0:
                unit_count["x"].append(cell.pos.x)
                unit_count["y"].append(cell.pos.y)
                unit_count["count"].append(str(ucount))
            if cell.citytile is not None:
                city["x"].append(cell.pos.x)
                city["y"].append(cell.pos.y)
                city["id"].append(cell.citytile.cityid)
                city["cooldown"].append(cell.citytile.cooldown)
                c = all_cities[cell.citytile.cityid]
                city["team"].append(c.team)
                city["fuel"].append(c.fuel)
                city["upkeep"].append(c.light_upkeep)
                if cityhighlight_function is not None:
                    if cell.citytile.team == state.id:
                        array = cityhighlight_function(state)
                        h = array[cell.pos.x, cell.pos.y]
                        highlight["x"].append(cell.pos.x)
                        highlight["y"].append(cell.pos.y)
                        highlight["level"].append(h)
            if cell.resource is not None:
                resource["x"].append(cell.pos.x)
                resource["y"].append(cell.pos.y)
                resource["type"].append(cell.resource.type)
                resource["amount"].append(cell.resource.amount)

    def common(name, data):
        return {
            "name": name,
            "x": data["x"],
            "y": data["y"],
            "text": get_tooltip(data),
            "mode": "markers",
        }

    traces.append(
        go.Scatter(
            **common("highlight", highlight),
            marker=dict(
                color="white",
                size=21,
                opacity=highlight["level"],
                symbol="square",
            ),
        )
    )
    traces.append(
        go.Scatter(
            **common("resource", resource),
            marker=dict(
                color=[resource_colors[t] for t in resource["type"]],
                size=16,
                opacity=0.8,
                symbol="square",
            ),
        )
    )
    traces.append(
        go.Scatter(
            **common("city", city),
            marker=dict(
                color=[colors[t] for t in city["team"]],
                colorscale="RdBu",
                cmax=1,
                cmin=0,
                size=18,
                opacity=0.6,
                symbol="square",
                line=dict(width=0, color="white"),
            ),
        )
    )
    traces.append(
        go.Scatter(
            **common("unit", unit),
            marker=dict(
                color=[colors[t] for t in unit["team"]],
                colorscale="RdBu",
                cmax=1,
                cmin=0,
                size=14,
                opacity=1,
                symbol="circle",
            ),
        )
    )
    traces.append(
        go.Scatter(
            name="unit_count",
            x=unit_count["x"],
            y=unit_count["y"],
            mode="text",
            text=unit_count["count"],
            textfont=dict(color="white", size=10),
            hoverinfo="skip",
        )
    )

    return traces


def make_figure(state, replay_id, player_id, widget=False):
    fig = make_subplots(
        rows=3,
        cols=2,
        specs=[
            [{"rowspan": 2}, {"rowspan": 2}],
            [None, None],
            [{"colspan": 2}, None],
        ],
        subplot_titles=(
            f"Episode: {replay_id}, step: {state.turn:03}",
            "Cell importance",
            f"Expected value (from {COLOR_NAMES[player_id]}'s perspective)",
        ),
    )
    if widget:
        fig = go.FigureWidget(fig)
    return fig


def update_figure_layout(fig, state):
    size = state.map.width - 0.5
    fig.update_xaxes(showgrid=False, zeroline=False, range=[-0.5, size], row=1, col=1)
    fig.update_xaxes(showgrid=False, zeroline=False, row=1, col=2)
    fig.update_xaxes(showgrid=False, zeroline=False, range=[0, 365], row=3, col=1)
    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        range=[size, -0.5],
        row=1,
        col=1,
    )
    fig.update_yaxes(showgrid=False, zeroline=False, autorange="reversed", row=1, col=2)
    fig.update_yaxes(showgrid=False, range=[-1, 1], row=3, col=1)
    fig.update_layout(
        showlegend=False,
        plot_bgcolor=DARK if is_night(state.turn) else LIGHT,
        autosize=False,
        width=(state.map.width * 30 + 5) * 2,
        height=(state.map.height * 30 + 5) + 150,
        margin=dict(l=10, r=10, b=10, t=30, pad=1),
    )
    fig.add_shape(
        type="rect",
        xref="x",
        yref="y",
        x0=0,
        y0=-1,
        x1=365,
        y1=1,
        line=dict(width=0),
        fillcolor=LIGHT,
        layer="below",
        row=3,
        col=1,
    )
    for x0, x1 in (
        (30, 40),
        (70, 80),
        (110, 120),
        (150, 160),
        (190, 200),
        (230, 240),
        (270, 280),
        (310, 320),
        (350, 360),
    ):
        fig.add_shape(
            type="rect",
            xref="x",
            yref="y",
            x0=x0,
            y0=-1,
            x1=x1,
            y1=1,
            line=dict(width=0),
            fillcolor=DARK,
            layer="below",
            row=3,
            col=1,
        )

    return fig


def add_traces(
    fig, state, heatmap_function, timeseries_function, cityhighlight_function
):
    traces = get_traces(state, cityhighlight_function)
    heatmap = get_heatmap_trace(state, heatmap_function)
    timeseries = get_timeseries_traces(state, timeseries_function)
    fig.add_traces(traces, rows=1, cols=1)
    fig.add_traces(heatmap, rows=1, cols=2)
    fig.add_traces(timeseries, rows=3, cols=1)
    fig = update_figure_layout(fig, state)


def get_zone_edges(state):
    a = state.zones.labels
    edges = list()
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            try:
                if a[i, j] != a[i + 1, j]:
                    edges.append(((i + 0.5, i + 0.5), (j - 0.5, j + 0.5)))
            except IndexError:
                pass
            try:
                if a[i, j] != a[i, j + 1]:
                    edges.append(((i - 0.5, i + 0.5), (j + 0.5, j + 0.5)))
            except IndexError:
                pass
    return edges


def plot_array(a, scale=0.4):
    a = a.T
    if a.ndim == 2:
        a = a.reshape((1,) + a.shape)
    for level, aa in enumerate(a):
        print(f"level {level}")
        I, J = [list(range(s)) for s in aa.shape]
        fig, ax = plt.subplots(
            figsize=(
                aa.shape[1] * scale + 0.5,
                aa.shape[0] * scale + 0.5,
            )
        )
        ax.imshow(aa, cmap="hot", interpolation="nearest")
        ax.set_xticks(J)
        ax.set_yticks(I)
        ax.set_xticklabels(J)
        ax.set_yticklabels(I)
        for i in I:
            for j in J:
                if np.issubdtype(a.dtype, np.floating):
                    label = f"{aa[i, j]:.1f}"
                else:
                    label = f"{aa[i, j]}"
                ax.text(j, i, label, ha="center", va="center", color="0.2")
        fig.tight_layout()
        plt.show()
