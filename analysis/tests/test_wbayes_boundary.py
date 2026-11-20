"""the *_noboundary variants exclude exactly the clamped rows (synthetic data only)."""
import math
import os
import random
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import fit_wbayes as fw  # noqa: E402
from common import logit_clipped  # noqa: E402


def build(entries):
    fw.EXCLUDED.clear()
    data, nb = {}, {}
    for i, (post, pre) in enumerate(entries):
        pri = random.Random(i).uniform(-2, 2)
        lr = math.log(5.0 if i % 2 else 0.2)
        fw.add_entry(data, nb, "m", "c", i // 2, pri, logit_clipped(pre / 100), lr,
                     logit_clipped(post / 100), post, pre)
    return data, nb


def nrows(d):
    return sum(len(v) for v in d.values())


class TestBoundary(unittest.TestCase):
    def test_excludes_exactly_boundary_rows(self):
        entries = [(30 + i, 20) for i in range(12)]
        entries[2] = (100, 20)   # boundary posttest
        entries[5] = (0, 20)     # boundary posttest
        entries[7] = (40, 0)     # boundary pretest only: excluded for "own" only
        data, nb = build(entries)
        k = lambda v: ("m", "c", v)
        self.assertEqual(nrows(data[k("published")]), 12)
        self.assertEqual(nrows(data[k("published_noboundary")]), 10)
        self.assertEqual(fw.EXCLUDED[k("published_noboundary")], 2)
        self.assertEqual(nrows(data[k("own")]), 12)
        self.assertEqual(nrows(data[k("own_noboundary")]), 9)
        self.assertEqual(fw.EXCLUDED[k("own_noboundary")], 3)
        self.assertEqual(nb[k("published")], 2)
        self.assertEqual(nb[k("own")], 3)
        self.assertEqual(nb[k("published_noboundary")], 0)
        self.assertEqual(nb[k("own_noboundary")], 0)

    def test_clean_data_variants_agree_exactly(self):
        entries = [(30 + 3 * i, 10 + i) for i in range(12)]
        data, nb = build(entries)
        for v in ("published", "own"):
            a = fw.fit_boot(data[("m", "c", v)], 50, random.Random(fw.SEED))
            b = fw.fit_boot(data[("m", "c", v + "_noboundary")], 50, random.Random(fw.SEED))
            self.assertEqual(a, b)
            self.assertEqual(fw.EXCLUDED.get(("m", "c", v + "_noboundary"), 0), 0)


    def _rows(self, main, nob, nbound=3):
        def row(v, d, n):
            r = {"model": "m", "condition": "c", "variant": v, "n_boundary_answers": n}
            for p in "ab":
                r[p], r[p + "_lo"], r[p + "_hi"] = d[p]
            return r
        return [row("published", main, nbound), row("published_noboundary", nob, 0)]

    def test_flag_rule(self):
        base = {"a": (1.0, 0.8, 1.2), "b": (1.5, 1.2, 1.8)}
        # no-boundary estimates inside the main CIs and vice versa: not flagged
        r = fw.boundary_flags(self._rows(base, {"a": (1.05, 0.7, 1.3), "b": (1.4, 1.1, 1.9)}))
        self.assertEqual([x["boundary_flag"] for x in r], [0, 0])
        # no-boundary b (1.1) outside the main CI [1.2, 1.8]: both rows flagged, parameter named
        r = fw.boundary_flags(self._rows(base, {"a": (1.0, 0.8, 1.2), "b": (1.1, 0.9, 1.7)}))
        self.assertEqual([x["boundary_flag"] for x in r], [1, 1])
        self.assertEqual(r[0]["boundary_flag_params"], "b")
        # main point outside the no-boundary CI also flags
        r = fw.boundary_flags(self._rows(base, {"a": (1.0, 0.9, 1.1), "b": (1.4, 1.0, 1.1)}))
        self.assertEqual(r[0]["boundary_flag_params"], "b")
        # no boundary answers in the cell: never flagged
        r = fw.boundary_flags(self._rows(base, {"a": (5.0, 4.0, 6.0), "b": (0.1, 0.0, 0.2)}, nbound=0))
        self.assertEqual([x["boundary_flag"] for x in r], [0, 0])


if __name__ == "__main__":
    unittest.main()
