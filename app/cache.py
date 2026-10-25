"""Small thread-safe TTL cache for whole answers (repeated questions return instantly)."""
import re
import threading
import time
from collections import OrderedDict


def normalize_question(q: str) -> str:
    """'What is Redis used for?' and 'what is redis used for' share one cache entry."""
    return re.sub(r"[^a-z0-9]+", " ", q.lower()).strip()


class TTLCache:
    def __init__(self, maxsize: int = 256, ttl_s: float = 3600):
        self.maxsize, self.ttl_s = maxsize, ttl_s
        self._data: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    def get(self, key):
        with self._lock:
            item = self._data.get(key)
            if item is None or time.monotonic() - item[0] > self.ttl_s:
                self._data.pop(key, None)
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return item[1]

    def set(self, key, value):
        with self._lock:
            self._data[key] = (time.monotonic(), value)
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def clear(self):
        with self._lock:
            self._data.clear()
