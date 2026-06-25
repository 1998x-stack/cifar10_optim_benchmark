# Visualization System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a real-time Dash browser dashboard + post-training static HTML report to the CIFAR-10 optimizer benchmark.

**Architecture:** New `visualization/` module with 4 files: thread-safe metrics store, Plotly figure builders, Dash dashboard (background daemon thread), and static HTML report generator. Trainer pushes metrics to the store each epoch; Dash polls the store every 2s via `dcc.Interval`. Same figure-building code drives both live and static output.

**Tech Stack:** Python 3, Dash, Plotly, Flask (Dash dependency), pandas (data wrangling), threading (stdlib), pytest (already available)

## Global Constraints

- Dash + Plotly must be installed via pip (not currently available)
- Dashboard starts automatically with `--compare`, optional with `--visualize` for single runs
- Dashboard runs in a daemon thread — stops when main process exits
- Port default 8050, auto-increment if occupied
- All figures use Plotly default color palette, optimizer-to-color mapping is consistent across figures
- Output reports saved under `logs/comparison/dashboard.html` or `logs/<experiment>/report.html`

---

### Task 1: Install dependencies

**Files:**
- Modify: `requirements.txt` (create if absent)

**Interfaces:**
- Consumes: nothing
- Produces: `dash`, `plotly` installed in Python environment

- [ ] **Step 1: Install dash and plotly**

```bash
pip install dash plotly
```

- [ ] **Step 2: Verify imports work**

```bash
python -c "import dash; import plotly; print(f'dash={dash.__version__}, plotly={plotly.__version__}')"
```

Expected: prints version numbers, no errors.

- [ ] **Step 3: Create requirements.txt**

```text
torch>=2.0
torchvision>=0.15
pyyaml
tensorboard
dash>=2.0
plotly>=5.0
pandas
```

File: `requirements.txt` at project root.

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "chore: add dash and plotly dependencies"
```

---

### Task 2: MetricsStore (thread-safe shared state)

**Files:**
- Create: `visualization/__init__.py`
- Create: `visualization/store.py`
- Test: `tests/visualization/test_store.py`

**Interfaces:**
- Produces: `MetricsStore` class with methods:
  - `__init__()` — creates empty store + `threading.Lock`
  - `ensure_optimizer(opt_name: str)` — initialize empty lists for an optimizer
  - `update(opt_name: str, **metrics: dict)` — append epoch metrics, thread-safe
  - `get(opt_name: str) -> dict` — return one optimizer's data (copy)
  - `get_all() -> dict` — return full store (copy under lock)
  - `set_status(opt_name: str, status: str)` — set "running"/"done"/"error"
  - `snapshot() -> dict` — deep copy of full store for report generation

- [ ] **Step 1: Write the failing test**

```python
# tests/visualization/test_store.py
import threading
from visualization.store import MetricsStore


def test_update_and_get():
    store = MetricsStore()
    store.ensure_optimizer("Adam")
    store.update("Adam", epoch=1, train_loss=0.5, train_acc=65.0)
    store.update("Adam", epoch=2, train_loss=0.3, train_acc=72.0)

    data = store.get("Adam")
    assert len(data["train"]) == 2
    assert data["train"][0] == {"epoch": 1, "train_loss": 0.5, "train_acc": 65.0}
    assert data["train"][1] == {"epoch": 2, "train_loss": 0.3, "train_acc": 72.0}


def test_get_returns_copy_not_reference():
    store = MetricsStore()
    store.ensure_optimizer("SGD")
    store.update("SGD", epoch=1, loss=0.5)

    copy1 = store.get("SGD")
    copy1["train"].append({"epoch": 2, "loss": 0.1})

    copy2 = store.get("SGD")
    assert len(copy2["train"]) == 1  # mutation to copy1 doesn't leak


def test_thread_safety():
    store = MetricsStore()
    store.ensure_optimizer("Adam")
    errors = []

    def push_epochs(start, end):
        for e in range(start, end):
            try:
                store.update("Adam", epoch=e, loss=float(e))
            except Exception as exc:
                errors.append(exc)

    t1 = threading.Thread(target=push_epochs, args=(0, 500))
    t2 = threading.Thread(target=push_epochs, args=(500, 1000))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert len(errors) == 0
    data = store.get("Adam")
    epochs = sorted(e["epoch"] for e in data["train"])
    assert epochs == list(range(1000))


def test_set_status():
    store = MetricsStore()
    store.ensure_optimizer("Adam")
    store.set_status("Adam", "running")
    assert store.get("Adam")["status"] == "running"
    store.set_status("Adam", "done")
    assert store.get("Adam")["status"] == "done"


def test_snapshot():
    store = MetricsStore()
    store.ensure_optimizer("Adam")
    store.update("Adam", epoch=1, loss=0.5)
    snap = store.snapshot()
    store.update("Adam", epoch=2, loss=0.3)
    # Snapshot is frozen at time of call
    assert len(snap["Adam"]["train"]) == 1
    assert len(store.get("Adam")["train"]) == 2
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/visualization/test_store.py -v
```

Expected: FAIL with ModuleNotFoundError for `visualization.store`.

- [ ] **Step 3: Write `visualization/__init__.py`**

```python
from .store import MetricsStore

__all__ = ["MetricsStore"]
```

- [ ] **Step 4: Write `visualization/store.py`**

```python
import copy
import threading


class MetricsStore:
    """Thread-safe per-optimizer metrics history."""

    def __init__(self):
        self._lock = threading.Lock()
        self._data: dict[str, dict] = {}

    def ensure_optimizer(self, opt_name: str) -> None:
        with self._lock:
            if opt_name not in self._data:
                self._data[opt_name] = {"train": [], "val": [], "status": "pending"}

    def update(self, opt_name: str, **metrics) -> None:
        with self._lock:
            if opt_name not in self._data:
                self._data[opt_name] = {"train": [], "val": [], "status": "pending"}
            row = metrics
            if metrics.get("phase") == "val":
                self._data[opt_name]["val"].append({k: v for k, v in row.items() if k != "phase"})
            else:
                self._data[opt_name]["train"].append({k: v for k, v in row.items() if k != "phase"})

    def get(self, opt_name: str) -> dict:
        with self._lock:
            return copy.deepcopy(self._data.get(opt_name, {"train": [], "val": [], "status": "unknown"}))

    def get_all(self) -> dict:
        with self._lock:
            return copy.deepcopy(self._data)

    def set_status(self, opt_name: str, status: str) -> None:
        with self._lock:
            if opt_name in self._data:
                self._data[opt_name]["status"] = status

    def snapshot(self) -> dict:
        return self.get_all()
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/visualization/test_store.py -v
```

Expected: 5 PASS.

- [ ] **Step 6: Commit**

```bash
git add visualization/__init__.py visualization/store.py tests/visualization/test_store.py
git commit -m "feat: add thread-safe MetricsStore for visualization"
```

---

### Task 3: Plotly figure builders

**Files:**
- Create: `visualization/figures.py`
- Test: `tests/visualization/test_figures.py`

**Interfaces:**
- Consumes: `MetricsStore.snapshot()` dict with shape `{opt_name: {train: [...], val: [...], status: str}}`
- Produces:
  - `build_val_acc_figure(snapshot: dict) -> go.Figure`
  - `build_train_loss_figure(snapshot: dict) -> go.Figure`
  - `build_best_acc_bar(snapshot: dict) -> go.Figure`
  - `build_throughput_chart(snapshot: dict) -> go.Figure`
  - `build_ranking_table(snapshot: dict) -> go.Figure`
  - `COLORS = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", ...]` — Plotly default palette

- [ ] **Step 1: Write the failing test**

```python
# tests/visualization/test_figures.py
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/visualization/test_figures.py -v
```

Expected: FAIL with ModuleNotFoundError or ImportError.

- [ ] **Step 3: Write `visualization/figures.py`**

```python
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
```

- [ ] **Step 4: Update `visualization/__init__.py`**

```python
from .store import MetricsStore
from . import figures

__all__ = ["MetricsStore", "figures"]
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/visualization/test_figures.py -v
```

Expected: 6 PASS.

- [ ] **Step 6: Commit**

```bash
git add visualization/__init__.py visualization/figures.py tests/visualization/test_figures.py
git commit -m "feat: add Plotly figure builders for dashboard"
```

---

### Task 4: Static HTML report generator

**Files:**
- Create: `visualization/report.py`
- Test: `tests/visualization/test_report.py`

**Interfaces:**
- Consumes: `build_*` functions from `figures.py`, `MetricsStore.snapshot()`
- Produces: `generate_report(snapshot: dict, output_path: str) -> None` — writes a standalone HTML file

- [ ] **Step 1: Write the failing test**

```python
# tests/visualization/test_report.py
import os
import tempfile
from visualization.report import generate_report


def make_snapshot():
    return {
        "Adam": {
            "train": [
                {"epoch": 1, "loss": 0.8, "acc": 60.0, "throughput": 120.0},
                {"epoch": 2, "loss": 0.5, "acc": 68.0, "throughput": 118.0},
            ],
            "val": [
                {"epoch": 1, "loss": 0.7, "acc": 62.0},
                {"epoch": 2, "loss": 0.45, "acc": 70.0},
            ],
            "status": "done",
            "test_acc": 69.0,
        },
    }


def test_generate_report_creates_html_file():
    snap = make_snapshot()
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        path = f.name
    try:
        generate_report(snap, path)
        assert os.path.exists(path)
        content = open(path).read()
        assert "<html" in content or "<HTML" in content or "plotly" in content
        assert os.path.getsize(path) > 1000  # non-trivial html
    finally:
        os.unlink(path)


def test_generate_report_empty_snapshot():
    snap = {}
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        path = f.name
    try:
        generate_report(snap, path)
        assert os.path.exists(path)
    finally:
        os.unlink(path)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/visualization/test_report.py -v
```

Expected: FAIL with ModuleNotFoundError.

- [ ] **Step 3: Write `visualization/report.py`**

```python
from datetime import datetime

from .figures import (
    build_val_acc_figure,
    build_train_loss_figure,
    build_best_acc_bar,
    build_throughput_chart,
    build_ranking_table,
)


def generate_report(snapshot: dict, output_path: str) -> None:
    """Generate a standalone HTML report from a metrics snapshot."""

    sections = []

    # Val accuracy
    fig1 = build_val_acc_figure(snapshot)
    sections.append(fig1.to_html(full_html=False, include_plotlyjs=False))

    # Training loss
    fig2 = build_train_loss_figure(snapshot)
    sections.append(fig2.to_html(full_html=False, include_plotlyjs=False))

    # Best accuracy bar
    fig3 = build_best_acc_bar(snapshot)
    sections.append(fig3.to_html(full_html=False, include_plotlyjs=False))

    # Throughput
    fig4 = build_throughput_chart(snapshot)
    sections.append(fig4.to_html(full_html=False, include_plotlyjs=False))

    # Ranking table
    fig5 = build_ranking_table(snapshot)
    sections.append(fig5.to_html(full_html=False, include_plotlyjs=False))

    # Assemble full HTML
    body = "\n".join(f'<div class="chart">{s}</div>' for s in sections)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CIFAR-10 Optimizer Benchmark Report</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         max-width: 1200px; margin: 0 auto; padding: 20px; background: #f8f9fa; }}
  h1 {{ color: #333; }}
  .chart {{ margin-bottom: 40px; background: white; padding: 20px; border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
</style>
</head>
<body>
<h1>CIFAR-10 Optimizer Benchmark</h1>
<p>Generated: {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")}</p>
{body}
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/visualization/test_report.py -v
```

Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add visualization/report.py tests/visualization/test_report.py
git commit -m "feat: add static HTML report generator"
```

---

### Task 5: Dash dashboard with background thread

**Files:**
- Create: `visualization/dashboard.py`

**Interfaces:**
- Consumes: `MetricsStore`, `figures.py` builders
- Produces:
  - `start_dashboard(store: MetricsStore, port: int = 8050, open_browser: bool = True) -> int` — starts Dash in daemon thread, returns actual port
  - `stop_dashboard()` — signals shutdown

- [ ] **Step 1: Write `visualization/dashboard.py`**

```python
import threading
import webbrowser
import time

from .store import MetricsStore
from .figures import (
    build_val_acc_figure,
    build_train_loss_figure,
    build_best_acc_bar,
    build_throughput_chart,
    build_ranking_table,
)

_dash_thread: threading.Thread | None = None
_store: MetricsStore | None = None


def start_dashboard(store: MetricsStore, port: int = 8050, open_browser: bool = True) -> int:
    """Start Dash dashboard in a background daemon thread. Returns the actual port."""
    global _dash_thread, _store
    _store = store

    from dash import Dash, dcc, html
    from dash.dependencies import Input, Output

    # Auto-increment port if occupied
    import socket
    while True:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(("127.0.0.1", port))
        sock.close()
        if result != 0:
            break
        port += 1

    app = Dash(__name__)

    app.layout = html.Div([
        html.H1("CIFAR-10 Optimizer Benchmark", style={"textAlign": "center"}),
        dcc.Interval(id="poll", interval=2000),
        html.Div(id="status-bar"),
        html.Div([
            dcc.Graph(id="val-acc-chart"),
            dcc.Graph(id="train-loss-chart"),
        ]),
        html.Div([
            dcc.Graph(id="best-acc-bar"),
            dcc.Graph(id="throughput-chart"),
        ]),
        html.Div([
            dcc.Graph(id="ranking-table"),
        ]),
    ], style={"maxWidth": "1200px", "margin": "0 auto", "fontFamily": "sans-serif"})

    @app.callback(
        [Output("val-acc-chart", "figure"),
         Output("train-loss-chart", "figure"),
         Output("best-acc-bar", "figure"),
         Output("throughput-chart", "figure"),
         Output("ranking-table", "figure"),
         Output("status-bar", "children")],
        [Input("poll", "n_intervals")]
    )
    def refresh(_n):
        snap = _store.get_all()
        # Build status cards
        status_parts = []
        for opt_name, data in snap.items():
            s = data.get("status", "?")
            color = {"running": "#2196F3", "done": "#4CAF50", "error": "#f44336"}.get(s, "#999")
            train_len = len(data.get("train", []))
            val_entries = data.get("val", [])
            best_val = max((e.get("acc", 0) for e in val_entries), default=0)
            status_parts.append(
                html.Div([
                    html.Strong(f"{opt_name} "),
                    html.Span(s, style={"color": color, "fontWeight": "bold"}),
                    html.Span(f" | Epochs: {train_len} | Best Val: {best_val:.1f}%"),
                ], style={"display": "inline-block", "margin": "5px 20px 5px 0"})
            )

        return (
            build_val_acc_figure(snap),
            build_train_loss_figure(snap),
            build_best_acc_bar(snap),
            build_throughput_chart(snap),
            build_ranking_table(snap),
            status_parts,
        )

    def _run():
        app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)

    _dash_thread = threading.Thread(target=_run, daemon=True)
    _dash_thread.start()

    # Give server a moment to start
    time.sleep(0.5)

    url = f"http://127.0.0.1:{port}"
    if open_browser:
        webbrowser.open(url)

    print(f"\n  Dashboard: {url}\n")

    return port


def stop_dashboard():
    """Signal shutdown. Daemon thread exits when main process exits."""
    pass
```

- [ ] **Step 2: Verify the module imports cleanly**

```bash
python -c "from visualization.dashboard import start_dashboard, stop_dashboard; print('OK')"
```

Expected: OK (Dash/Flask import may fail if not installed, re-run Task 1 if needed).

- [ ] **Step 3: Smoke test — start dashboard, verify it doesn't crash**

```bash
python -c "
from visualization.store import MetricsStore
from visualization.dashboard import start_dashboard
store = MetricsStore()
store.ensure_optimizer('Adam')
store.update('Adam', epoch=1, loss=0.8, acc=60.0, throughput=120.0, phase='train')
store.update('Adam', epoch=1, loss=0.7, acc=62.0, phase='val')
port = start_dashboard(store, port=8060, open_browser=False)
print(f'Dashboard started on port {port}')
import time; time.sleep(2)
print('Dashboard running OK')
"
```

Expected: prints "Dashboard started on port 8060" and "Dashboard running OK". Manually visit `http://127.0.0.1:8060` to verify charts render.

- [ ] **Step 4: Commit**

```bash
git add visualization/dashboard.py
git commit -m "feat: add Dash dashboard with background thread"
```

---

### Task 6: Integration — main.py

**Files:**
- Modify: `main.py`

**Interfaces:**
- Consumes: `MetricsStore`, `start_dashboard`, `stop_dashboard`, `generate_report`
- Produces: `--visualize`, `--no-visualize`, `--report LOG_DIR` CLI flags; --compare auto-starts dashboard

- [ ] **Step 1: Read current `main.py` and apply changes**

```python
#!/usr/bin/env python3
"""
CIFAR-10 Optimizer Benchmark
...
Usage:
    # Single optimizer
    python main.py optimizer.name=Adam

    # Single optimizer with live dashboard
    python main.py optimizer.name=Adam --visualize

    # Comparison (all 4 optimizers) — auto-starts dashboard
    python main.py --compare

    # Comparison without dashboard
    python main.py --compare --no-visualize

    # Backfill: generate report from historical JSON
    python main.py --report logs/ResNet18_Adam/
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import json

from core.config_loader import ConfigLoader
from experiments.run_single import run_single_experiment
from experiments.run_comparison import run_comparison
from visualization.store import MetricsStore
from visualization.dashboard import start_dashboard, stop_dashboard
from visualization.report import generate_report


def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 Optimizer Benchmark")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    parser.add_argument(
        "--compare", action="store_true",
        help="Run comparison across all optimizers (Adam, AdamW, SGD, RMSprop)",
    )
    parser.add_argument(
        "--optimizers", type=str, nargs="+", default=None,
        help="Optimizers to compare (default: Adam AdamW SGD RMSprop)",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Enable live Dash dashboard (auto-enabled with --compare)",
    )
    parser.add_argument(
        "--no-visualize", action="store_true",
        help="Disable dashboard even with --compare",
    )
    parser.add_argument(
        "--report", type=str, default=None,
        help="Generate static HTML report from existing experiment log dir",
    )
    parser.add_argument(
        "overrides", nargs="*",
        help="CLI overrides: optimizer.name=AdamW training.epochs=100",
    )
    args = parser.parse_args()

    # ── Backfill mode ──
    if args.report:
        _generate_backfill_report(args.report)
        return

    # Parse overrides
    cli_overrides = {}
    for item in args.overrides:
        if "=" in item:
            key, value = item.split("=", 1)
            cli_overrides[key] = value

    # Load config
    loader = ConfigLoader(args.config)
    config = loader.load(cli_overrides)

    # Dashboard: auto-enable for comparison, unless --no-visualize
    use_dashboard = (args.compare or args.visualize) and not args.no_visualize

    # Print config
    print("=" * 60)
    print("  CIFAR-10 Optimizer Benchmark")
    print("=" * 60)
    opt_name = config.get("optimizer", {}).get("name", "Adam")
    model_name = config.get("model", {}).get("name", "ResNet18")
    epochs = config.get("training", {}).get("epochs", 50)
    print(f"  Model:    {model_name}")
    print(f"  Optimizer: {opt_name if not args.compare else 'COMPARISON'}")
    print(f"  Epochs:   {epochs}")
    print(f"  Device:   {config.get('experiment', {}).get('device', 'auto')}")
    print(f"  Dashboard: {'Yes' if use_dashboard else 'No'}")
    print("=" * 60)

    # Start dashboard if requested
    store = None
    dashboard_port = None
    if use_dashboard:
        store = MetricsStore()
        dashboard_port = start_dashboard(store, open_browser=True)

    if args.compare:
        optimizers = args.optimizers or ["Adam", "AdamW", "SGD", "RMSprop"]
        print(f"\n  Comparing optimizers: {', '.join(optimizers)}")
        results = run_comparison(config, optimizers=optimizers, store=store)
        # Generate report
        if store is not None:
            from pathlib import Path
            log_dir = Path(config.get("logging", {}).get("log_dir", "./logs"))
            report_path = log_dir / "comparison" / "dashboard.html"
            # Enrich snapshot with final test results
            snap = store.snapshot()
            for opt_name, summary in results.items():
                if opt_name in snap:
                    snap[opt_name]["test_acc"] = summary.get("test_acc", "N/A")
            generate_report(snap, str(report_path))
            print(f"\n  Report saved: {report_path}")
    else:
        summary = run_single_experiment(config, store=store)
        print(f"\n  Single run complete: {summary.get('experiment', '?')}")
        print(f"  Best Val Acc: {summary.get('best_val_acc', 0):.2f}%")
        print(f"  Test Acc:     {summary.get('test_acc', 0):.2f}%")
        # Generate report for single run
        if store is not None:
            from pathlib import Path
            log_dir = Path(config.get("logging", {}).get("log_dir", "./logs"))
            exp_name = f"{model_name}_{opt_name}"
            report_path = log_dir / exp_name / "report.html"
            snap = store.snapshot()
            if exp_name in snap:
                snap[exp_name]["test_acc"] = summary.get("test_acc", "N/A")
            generate_report(snap, str(report_path))
            print(f"  Report saved: {report_path}")

    # Keep alive if dashboard running so user can view charts
    if dashboard_port is not None:
        print(f"\n  Dashboard running at http://127.0.0.1:{dashboard_port}")
        print("  Press Ctrl+C to exit.\n")
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n  Shutting down.")


def _generate_backfill_report(log_dir: str):
    """Generate a static HTML report from an existing JSON metrics file."""
    from pathlib import Path
    p = Path(log_dir)
    metrics_file = p / "metrics_history.json"
    if not metrics_file.exists():
        print(f"Error: {metrics_file} not found")
        sys.exit(1)
    metrics = json.loads(metrics_file.read_text())
    # Convert per-epoch metrics list to MetricsStore format
    store = MetricsStore()
    exp_name = p.name  # e.g. "ResNet18_Adam"
    store.ensure_optimizer(exp_name)
    for entry in metrics:
        epoch = entry.get("step", 0)
        store.update(exp_name, epoch=epoch, **{k: v for k, v in entry.items() if k != "step"})
    store.set_status(exp_name, "done")
    report_path = p / "report.html"
    generate_report(store.snapshot(), str(report_path))
    print(f"Report saved: {report_path}")
    sys.exit(0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify main.py imports cleanly**

```bash
python -c "import main; print('OK')"
```

Expected: OK.

- [ ] **Step 3: Commit**

```bash
git add main.py
git commit -m "feat: integrate dashboard and report into main.py"
```

---

### Task 7: Integration — Trainer + experiment runner

**Files:**
- Modify: `trainers/cifar_trainer.py`
- Modify: `experiments/run_single.py`
- Modify: `experiments/run_comparison.py`

**Interfaces:**
- Consumes: `MetricsStore` from Task 2
- Produces: `CIFARTrainer` accepts optional `store` param; `run_single_experiment` passes it through; `run_comparison` pushes progress

- [ ] **Step 1: Modify `trainers/cifar_trainer.py` — accept optional store**

Add `store` parameter to `__init__` and push metrics in `fit()` method.

The key changes:

In `__init__`, add:
```python
def __init__(
    self,
    model, optimizer_wrapper, scheduler, device, exp_logger, config,
    criterion=None,
    store=None,            # <-- added
    experiment_name=None,  # <-- added
):
    super().__init__(model, device, exp_logger, config)
    ...
    self.store = store
    self.experiment_name = experiment_name or "experiment"
```

In `fit()`, after `self.train()` returns metrics, push to store:

```python
def fit(self, train_loader, val_loader, test_loader=None, num_epochs=50):
    self.train_loader = train_loader
    self.val_loader = val_loader
    metrics = self.train(train_loader, val_loader, num_epochs)

    # Push to visualization store
    if self.store is not None:
        self.store.ensure_optimizer(self.experiment_name)
        for train_entry in metrics.get("train", []):
            self.store.update(self.experiment_name, phase="train", **train_entry)
        for val_entry in metrics.get("val", []):
            self.store.update(self.experiment_name, phase="val", **val_entry)

    if test_loader is not None:
        test_metrics = self.test(test_loader)
        metrics["test"] = test_metrics
        self.log.log_metrics(test_metrics, self.current_epoch, prefix="test")
        self.log.info(...)
        if self.store is not None:
            self.store.update(self.experiment_name, phase="test", **test_metrics)

    if self.store is not None:
        self.store.set_status(self.experiment_name, "done")

    return metrics
```

- [ ] **Step 2: Modify `experiments/run_single.py` — pass store through**

In `run_single_experiment(config, store=None)`:

```python
def run_single_experiment(config: dict, store=None) -> dict:
    ...
    exp_name = (
        f"{config.get('model', {}).get('name', 'ResNet18')}"
        f"_{config.get('optimizer', {}).get('name', 'Adam')}"
    )

    trainer = CIFARTrainer(
        model=model,
        optimizer_wrapper=optimizer_wrapper,
        scheduler=scheduler,
        device=device,
        exp_logger=exp_logger,
        config=config,
        store=store,             # <-- pass through
        experiment_name=exp_name,  # <-- pass through
    )
    ...
```

- [ ] **Step 3: Modify `experiments/run_comparison.py` — push progress to store**

In `run_comparison(base_config, optimizers=None, store=None)`:

```python
def run_comparison(
    base_config: dict,
    optimizers: list[str] | None = None,
    store=None,  # <-- added
) -> dict[str, dict]:
    ...
    for opt_name in optimizers:
        ...
        config["optimizer"]["name"] = opt_name
        if store is not None:
            store.ensure_optimizer(opt_name)
            store.set_status(opt_name, "running")

        try:
            summary = run_single_experiment(config)
            results[opt_name] = summary
            if store is not None:
                store.set_status(opt_name, "done")
        except Exception as e:
            if store is not None:
                store.set_status(opt_name, "error")
            raise

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    ...
```

- [ ] **Step 4: Verify imports**

```bash
python -c "
from trainers.cifar_trainer import CIFARTrainer
from experiments.run_single import run_single_experiment
from experiments.run_comparison import run_comparison
from visualization.store import MetricsStore
print('All imports OK')
"
```

Expected: All imports OK.

- [ ] **Step 5: Commit**

```bash
git add trainers/cifar_trainer.py experiments/run_single.py experiments/run_comparison.py
git commit -m "feat: thread metrics store through trainer and experiment runners"
```

---

### Task 8: End-to-end smoke test

**Files:**
- Test: `tests/visualization/test_integration.py`

- [ ] **Step 1: Write integration test (short training run, verify store populated)**

```python
# tests/visualization/test_integration.py
import tempfile
import os
from pathlib import Path
from visualization.store import MetricsStore
from visualization.report import generate_report


def test_store_to_report_pipeline():
    """End-to-end: populate store, generate report, verify file exists."""
    store = MetricsStore()
    store.ensure_optimizer("Adam")
    store.ensure_optimizer("SGD")
    for e in range(1, 4):
        store.update("Adam", epoch=e, loss=1.0 / e, acc=60 + e * 5,
                     throughput=100 + e, phase="train")
        store.update("Adam", epoch=e, loss=0.9 / e, acc=58 + e * 5,
                     phase="val")
        store.update("SGD", epoch=e, loss=1.5 / e, acc=50 + e * 4,
                     throughput=95 + e, phase="train")
    store.set_status("Adam", "done")
    store.set_status("SGD", "done")

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        path = f.name
    try:
        generate_report(store.snapshot(), path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 2000
        content = open(path).read()
        assert "Adam" in content
        assert "SGD" in content
    finally:
        os.unlink(path)
```

- [ ] **Step 2: Run integration test**

```bash
python -m pytest tests/visualization/test_integration.py -v
```

Expected: 1 PASS.

- [ ] **Step 3: Run all visualization tests**

```bash
python -m pytest tests/visualization/ -v
```

Expected: all tests PASS (14 total: 5 store + 6 figures + 2 report + 1 integration).

- [ ] **Step 4: Commit**

```bash
git add tests/visualization/test_integration.py
git commit -m "test: add end-to-end store-to-report pipeline test"
```
