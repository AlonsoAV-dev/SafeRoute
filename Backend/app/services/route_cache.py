from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import Future
from copy import deepcopy
from threading import RLock
from time import monotonic


# Conserva consultas recientes con vencimiento y coordina solicitudes simultáneas equivalentes.
class RouteComparisonCache:
    """Resultados exactos por parámetros; acotados y aislados de quien los lee."""

    def __init__(self, max_entries=64, ttl_seconds=300, clock=monotonic):
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.clock = clock
        self._entries = OrderedDict()
        self._pending = {}
        self._lock = RLock()

    # Reutiliza un resultado válido o ejecuta una sola consulta compartida entre quienes la esperan.
    def get_or_compute(self, key, compute):
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and self.clock() - entry[0] < self.ttl_seconds:
                self._entries.move_to_end(key)
                snapshot, future, owner = entry[1], None, False
            else:
                self._entries.pop(key, None)
                snapshot = None
                future = self._pending.get(key)
                owner = future is None
                if owner:
                    future = Future()
                    self._pending[key] = future
        if snapshot is not None:
            return deepcopy(snapshot), True
        if not owner:
            return deepcopy(future.result()), True
        try:
            result = compute()
            snapshot = deepcopy(result)
            with self._lock:
                self._entries[key] = (self.clock(), snapshot)
                self._entries.move_to_end(key)
                while len(self._entries) > self.max_entries:
                    self._entries.popitem(last=False)
                self._pending.pop(key, None)
                future.set_result(snapshot)
            return result, False
        except BaseException as error:
            with self._lock:
                self._pending.pop(key, None)
                future.set_exception(error)
            raise
