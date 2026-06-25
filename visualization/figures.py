import plotly.graph_objects as go

COLORS = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
]

_OPT_COLOR_MAP: dict[str, str] = {}
_color_idx = 0


def _color_for(opt_name: str) -> str:
    global _color_idx
    if opt_name not in _OPT_COLOR_MAP:
        _OPT_COLOR_MAP[opt_name] = COLORS[_color_idx % len(COLORS)]
        _color_idx += 1
    return _OPT_COLOR_MAP[opt_name]


def build_val_acc_figure(snapshot: dict) -> go.Figure:
    fig = go.Figure()
    for opt_name, data in snapshot.items():
        val_entries = data.get("val", [])
        if not val_entries:
            continue
        epochs = [e["epoch"] for e in val_entries]
        accs = [e["acc"] for e in val_entries]
        fig.add_trace(go.Scatter(
            x=epochs, y=accs, mode="lines+markers",
            name=opt_name, line=dict(color=_color_for(opt_name)),
        ))
    fig.update_layout(
        title="Validation Accuracy",
        xaxis_title="Epoch",
        yaxis_title="Accuracy (%)",
        hovermode="x unified",
    )
    return fig


def build_train_loss_figure(snapshot: dict) -> go.Figure:
    fig = go.Figure()
    for opt_name, data in snapshot.items():
        train_entries = data.get("train", [])
        if not train_entries:
            continue
        epochs = [e["epoch"] for e in train_entries]
        losses = [e["loss"] for e in train_entries]
        fig.add_trace(go.Scatter(
            x=epochs, y=losses, mode="lines+markers",
            name=opt_name, line=dict(color=_color_for(opt_name)),
        ))
    fig.update_layout(
        title="Training Loss",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        hovermode="x unified",
    )
    return fig


def build_best_acc_bar(snapshot: dict) -> go.Figure:
    names = []
    best_vals = []
    for opt_name, data in snapshot.items():
        val_entries = data.get("val", [])
        best = max((e["acc"] for e in val_entries), default=0.0)
        names.append(opt_name)
        best_vals.append(round(best, 2))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names, y=best_vals,
        marker_color=[_color_for(n) for n in names],
        text=best_vals, textposition="outside",
    ))
    fig.update_layout(
        title="Best Validation Accuracy",
        xaxis_title="Optimizer",
        yaxis_title="Accuracy (%)",
    )
    return fig


def build_throughput_chart(snapshot: dict) -> go.Figure:
    names = []
    throughputs = []
    for opt_name, data in snapshot.items():
        train_entries = data.get("train", [])
        avg = 0.0
        if train_entries:
            avg = sum(e.get("throughput", 0.0) for e in train_entries) / len(train_entries)
        names.append(opt_name)
        throughputs.append(round(avg, 1))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=names, x=throughputs, orientation="h",
        marker_color=[_color_for(n) for n in names],
        text=throughputs, textposition="outside",
    ))
    fig.update_layout(
        title="Training Throughput (samples/sec)",
        xaxis_title="Throughput",
    )
    return fig


def build_ranking_table(snapshot: dict) -> go.Figure:
    rows = []
    for opt_name, data in snapshot.items():
        val_entries = data.get("val", [])
        best_val = max((e["acc"] for e in val_entries), default=0.0)
        test_acc = data.get("test_acc", "N/A")
        rows.append((opt_name, round(best_val, 2), test_acc))

    rows.sort(key=lambda r: r[1], reverse=True)
    for i, row in enumerate(rows):
        rows[i] = (i + 1,) + row

    fig = go.Figure()
    fig.add_trace(go.Table(
        header=dict(values=["Rank", "Optimizer", "Best Val Acc (%)", "Test Acc (%)"]),
        cells=dict(values=[[r[0] for r in rows], [r[1] for r in rows],
                           [r[2] for r in rows], [r[3] for r in rows]]),
    ))
    return fig
