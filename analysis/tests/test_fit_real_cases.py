"""Exact-Bayes answers on the real case files must be recovered by the fit as a=b=1, c=0."""
import math
import os
import random
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import common  # noqa: E402
import fit_wbayes  # noqa: E402

CASES = common.load_cases(os.path.join(common.ROOT, "data", "cases"))
FIT = {k: c for k, c in CASES.items() if not c.get("exclude_from_fit")}


def clusters(fn):
    cl = {}
    for c in FIT.values():
        t = c["target"]
        xp = common.logit_clipped(t["pretest_point"])
        xl = math.log(t["lr_point"])
        cl.setdefault(c["scenario"], []).append((xp, xl, fn(c, xp, xl)))
    return cl


class RealCaseFit(unittest.TestCase):
    def test_fit_set_nonempty_and_targets_unclipped(self):
        self.assertGreater(len(FIT), 10)
        for k, c in FIT.items():
            common.check_unclipped(c["target"]["pretest_point"], k)
            common.check_unclipped(c["target"]["posttest_point"], k)

    def test_exact_targets_match_prior_times_lr(self):
        for k, c in FIT.items():
            t = c["target"]
            got = common.logit_clipped(t["posttest_point"])
            want = common.logit_clipped(t["pretest_point"]) + math.log(t["lr_point"])
            self.assertAlmostEqual(got, want, places=6, msg=k)

    def test_exact_bayes_recovers_a1_b1_c0(self):
        r = fit_wbayes.fit_boot(clusters(lambda c, xp, xl:
                                         common.logit_clipped(c["target"]["posttest_point"])),
                                20, random.Random(1))
        for name, want in (("a", 1.0), ("b", 1.0), ("c", 0.0)):
            self.assertAlmostEqual(r[name], want, delta=1e-6, msg=str(r))
        print("recovered a,b,c =", r["a"], r["b"], r["c"])

    def test_synthetic_a07_b08_recovered(self):
        r = fit_wbayes.fit_boot(clusters(lambda c, xp, xl: 0.7 * xp + 0.8 * xl), 20,
                                random.Random(1))
        self.assertAlmostEqual(r["a"], 0.7, delta=1e-6)
        self.assertAlmostEqual(r["b"], 0.8, delta=1e-6)
        self.assertAlmostEqual(r["c"], 0.0, delta=1e-6)

    def test_boundary_flag(self):
        self.assertTrue(common.is_boundary(0.0))
        self.assertTrue(common.is_boundary(1.0))
        self.assertFalse(common.is_boundary(0.0005))
        self.assertTrue(math.isfinite(common.logit_clipped(0.0)))


if __name__ == "__main__":
    unittest.main()
