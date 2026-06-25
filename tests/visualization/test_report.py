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
