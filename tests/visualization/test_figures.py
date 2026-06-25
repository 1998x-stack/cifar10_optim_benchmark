import plotly.graph_objects as go
from visualization.figures import (
    build_val_acc_figure,
    build_train_loss_figure,
    build_best_acc_bar,
    build_ranking_table,
)


def make_snapshot():
    return {
        "Adam": {
            "train": [
                {"epoch": 1, "loss": 0.8, "acc": 60.0},
                {"epoch": 2, "loss": 0.5, "acc": 68.0},
            ],
            "val": [
                {"epoch": 1, "loss": 0.7, "acc": 62.0},
                {"epoch": 2, "loss": 0.45, "acc": 70.0},
            ],
            "status": "done",
        },
        "SGD": {
            "train": [
                {"epoch": 1, "loss": 1.2, "acc": 50.0},
                {"epoch": 2, "loss": 0.9, "acc": 58.0},
            ],
            "val": [
                {"epoch": 1, "loss": 1.1, "acc": 52.0},
                {"epoch": 2, "loss": 0.85, "acc": 60.0},
            ],
            "status": "done",
        },
    }


def test_build_val_acc_figure_returns_figure():
    snap = make_snapshot()
    fig = build_val_acc_figure(snap)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2  # two traces, one per optimizer
    assert fig.data[0].mode == "lines+markers"


def test_build_val_acc_figure_handles_empty_val():
    snap = {
        "Adam": {"train": [], "val": [], "status": "running"},
    }
    fig = build_val_acc_figure(snap)
    assert isinstance(fig, go.Figure)


def test_build_train_loss_returns_figure():
    snap = make_snapshot()
    fig = build_train_loss_figure(snap)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2


def test_build_best_acc_bar_returns_figure():
    snap = make_snapshot()
    fig = build_best_acc_bar(snap)
    assert isinstance(fig, go.Figure)
    # One bar per optimizer
    assert len(fig.data) == 1
    assert len(fig.data[0].x) == 2  # Adam, SGD


def test_build_best_acc_bar_handles_empty():
    snap = {"Adam": {"train": [], "val": [], "status": "running"}}
    fig = build_best_acc_bar(snap)
    assert isinstance(fig, go.Figure)


def test_build_ranking_table_returns_figure():
    snap = make_snapshot()
    fig = build_ranking_table(snap)
    assert isinstance(fig, go.Figure)
    # Table has headers and cells
    assert len(fig.data[0].header.values) > 0
