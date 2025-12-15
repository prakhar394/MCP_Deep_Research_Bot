import time
from pathlib import Path
from typing import Any, Dict, Optional


class NamespacedCache:
    def __init__(self, base_dir: str = ".cache"):
        self._mem: Dict[str, Dict[str, Any]] = {}
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)

    def _ns_key(self, namespace: str, key: str) -> str:
        return f"{namespace}:{key}"

    def get(self, namespace: str, key: str) -> Optional[Any]:
        full = self._ns_key(namespace, key)
        item = self._mem.get(full)
        if not item:
            return None
        expires_at = item["expires_at"]
        if expires_at and expires_at < time.time():
            self._mem.pop(full, None)
            return None
        return item["value"]

    def set(self, namespace: str, key: str, value: Any, ttl: int = 3600):
        full = self._ns_key(namespace, key)
        self._mem[full] = {
            "value": value,
            "expires_at": time.time() + ttl if ttl > 0 else None,
        }


_cache = NamespacedCache()


def get_cache() -> NamespacedCache:
    return _cache
