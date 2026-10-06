import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Event

from app.services.route_cache import RouteComparisonCache


class RouteCacheTests(unittest.TestCase):
    def test_repeat_is_reused_without_sharing_mutable_results(self):
        cache = RouteComparisonCache()
        first, hit = cache.get_or_compute(("rf", "2026-09"), lambda: {"segments": [{"red": 0}]})
        self.assertFalse(hit)
        first["segments"][0]["red"] = 99
        second, hit = cache.get_or_compute(("rf", "2026-09"), lambda: self.fail("Repeated calculation"))
        self.assertTrue(hit)
        self.assertEqual(second["segments"][0]["red"], 0)
        second["segments"].clear()
        third, _ = cache.get_or_compute(("rf", "2026-09"), lambda: self.fail("Repeated calculation"))
        self.assertEqual(len(third["segments"]), 1)

    def test_expired_or_different_period_and_model_recalculates(self):
        now = [0.]
        cache = RouteComparisonCache(ttl_seconds=5, clock=lambda: now[0])
        cache.get_or_compute(("rf", "2026-09"), lambda: {"score": 1})
        now[0] = 6
        self.assertEqual(cache.get_or_compute(("rf", "2026-09"), lambda: {"score": 2}), ({"score": 2}, False))
        self.assertFalse(cache.get_or_compute(("rf", "2026-10"), lambda: {"score": 3})[1])
        self.assertFalse(cache.get_or_compute(("xgb", "2026-09"), lambda: {"score": 4})[1])

    def test_capacity_is_bounded(self):
        cache = RouteComparisonCache(max_entries=1)
        cache.get_or_compute("old", lambda: {})
        cache.get_or_compute("new", lambda: {})
        self.assertFalse(cache.get_or_compute("old", lambda: {})[1])

    def test_failed_request_can_be_retried(self):
        cache = RouteComparisonCache()
        def fail():
            raise ValueError("No connected route")
        with self.assertRaises(ValueError):
            cache.get_or_compute("route", fail)
        self.assertEqual(cache.get_or_compute("route", lambda: {"ok": True}), ({"ok": True}, False))

    def test_simultaneous_identical_requests_compute_once(self):
        cache = RouteComparisonCache()
        started, release = Event(), Event()
        calls = []
        def compute():
            calls.append(1)
            started.set()
            if not release.wait(2):
                raise TimeoutError()
            return {"red": 0}
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(cache.get_or_compute, "route", compute)
            self.assertTrue(started.wait(2))
            second = pool.submit(cache.get_or_compute, "route", compute)
            release.set()
            self.assertEqual(first.result()[0], second.result()[0])
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
