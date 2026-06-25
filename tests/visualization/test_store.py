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
