"""Exact Bayesian targets for the cases. Every target in data/cases/ is computed here, never by hand."""
import math


def logit(p):
    if not 0.0 < p < 1.0:
        raise ValueError(f"probability must be in (0, 1), got {p}")
    return math.log(p / (1.0 - p))


def expit(x):
    return 1.0 / (1.0 + math.exp(-x))


def posterior(prior, lrs):
    """Posterior probability from a prior and a list of likelihood ratios.

    Combining more than one LR assumes conditional independence of the findings given disease status.
    """
    log_odds = logit(prior) + sum(math.log(lr) for lr in lrs)
    return expit(log_odds)


if __name__ == "__main__":
    # Self-checks: a prior of 0.5 with LR 1 stays 0.5; prior 0.2 with LR 4 gives odds 1, so 0.5.
    assert abs(posterior(0.5, [1.0]) - 0.5) < 1e-12
    assert abs(posterior(0.2, [4.0]) - 0.5) < 1e-12
    # LR+ 9 and LR- 0.1 on a 10% prior (textbook example): 0.5 and about 0.011.
    assert abs(posterior(0.1, [9.0]) - 0.5) < 1e-12
    assert abs(posterior(0.1, [0.1]) - (1 / 91)) < 1e-12
    print("bayes.py self-checks passed")
