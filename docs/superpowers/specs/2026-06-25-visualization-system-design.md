# Visualization System Design

## Goal

Add a real-time browser dashboard + post-training HTML report to the CIFAR-10 optimizer benchmark. Training starts a Dash server automatically; the same Plotly figure code drives both live and static output.

## New Module: `visualization/`

```
visualization/
├── __init__.py
├── store.py          # Thread-safe metrics store
├── figures.py        # Plotly figure builders
├── dashboard.py      # Dash app + background thread
└── report.py         # Static HTML report generation
```

## Data Flow

```
  trainer thread            Dash thread (background)
       │                          │
       │ push each epoch          │ dcc.Interval polls every 2s
       ▼                          ▼
           MetricsStore (threading.Lock)
           {opt_name: {train: [...], val: [...], status, config, ...}}
                    │
                    ▼
             figures.py  ←── shared figure-building functions
            /           \
           /             \
   dashboard.py       report.py
   (live Dash)        (static HTML)
```

## Components

### `store.py` — MetricsStore

Thread-safe dict wrapper with `threading.Lock`. API:
- `update(opt_name, data)`: merge new metrics (called from trainer)
- `get(opt_name) → dict`: get full history for one optimizer
- `get_all() → dict`: get everything (Dash reads this)
- `set_status(opt_name, status)`: "running" / "done" / "error"
- `snapshot() → dict`: deep copy for report generation

### `figures.py` — Plotly Figure Builders

Each function takes a MetricsStore snapshot, returns a `plotly.graph_objects.Figure`:
- `build_val_acc_figure(snapshot)` — multiline chart, one trace per optimizer
- `build_train_loss_figure(snapshot)` — same layout
- `build_best_acc_bar(snapshot)` — grouped bar, one bar per optimizer
- `build_throughput_chart(snapshot)` — horizontal bar
- `build_ranking_table(snapshot)` — Plotly Table widget

Color palette: Plotly default qualitative palette (10 colors), assigned per optimizer name consistently across all figures.

### `dashboard.py` — Dash App

- Single Dash app with `dcc.Interval(id="poll", interval=2000)` for refresh
- All chart components defined once; callbacks rebuild figures from store on each interval tick
- `start_dashboard(store, port=8050, open_browser=True)`: starts Flask+Dash in a daemon thread, prints local URL
- `stop_dashboard()`: signals shutdown, joins thread
- Port: default 8050, auto-increment if occupied (8050, 8051, 8052...)

Page layout (single page, top to bottom):

| Section      | Contents                                                |
|-------------|---------------------------------------------------------|
| Status bar  | Progress indicators, current metrics cards per optimizer |
| Row 1       | Val Accuracy curve (all optimizers)                      |
| Row 2       | Training Loss curve (all optimizers)                     |
| Row 3       | Best Accuracy bar chart + Throughput bar chart           |
| Row 4       | Ranking summary table                                    |

### `report.py` — Static HTML Report

- `generate_report(snapshot, output_path)`: uses the same `figures.py` builders, writes a standalone HTML file with Plotly.js embedded (via `plotly.offline.plot` or `figure.write_html()`)
- Output path: `logs/comparison/dashboard.html` for comparison, `logs/<experiment>/report.html` for single runs
- Generated automatically after training completes

### Integration Points

**`main.py` changes:**
- `--visualize` flag: enable dashboard (auto-enabled when `--compare` is used)
- `--no-visualize` flag: explicitly disable
- `--report LOG_DIR`: backfill mode — generate static report from existing JSON

**`run_comparison.py` changes:**
- Accept optional `store` parameter
- Push metrics + status to store after each optimizer run

**`run_single.py` / `CIFARTrainer` changes:**
- Accept optional `store` parameter
- Push train/val metrics after each epoch via a callback or direct call

Trigger: `python main.py --compare` → creates store → starts dashboard → runs comparison → generates report → waits for user to press Ctrl+C (or auto-exits with `--exit-after`)

## Dependencies

Added to project: `dash`, `plotly`, `flask` (Dash depends on it), `pandas` (data wrangling for figures).

## Non-Goals

- Multi-user support (single local user)
- Authentication or access control
- Persisting dashboard state across restarts
- GPU memory profiling
- Model architecture visualization (already handled by TensorBoard via `log_model_graph`)
