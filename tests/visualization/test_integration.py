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
