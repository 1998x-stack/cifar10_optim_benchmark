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
