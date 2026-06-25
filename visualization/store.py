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
