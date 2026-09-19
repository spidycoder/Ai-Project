"""Statistics used to decide (a) whether a drop in score is a real
regression or just noise, and (b) whether the LLM judge agrees with human
graders enough to be trusted. Written with plain loops and the standard
library only (random, statistics) - no numpy."""
import random
import statistics


class RegressionResult(object):
    def __init__(self, baseline_mean, candidate_mean, ci_low, ci_high, is_regression):
        self.baseline_mean = baseline_mean
        self.candidate_mean = candidate_mean
        self.delta = candidate_mean - baseline_mean
        self.ci_low = ci_low     # low end of the 95% confidence interval on the delta
        self.ci_high = ci_high   # high end of the 95% confidence interval on the delta
        self.is_regression = is_regression

    def as_dict(self):
        return {
            "baseline_mean": self.baseline_mean,
            "candidate_mean": self.candidate_mean,
            "delta": self.delta,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "is_regression": self.is_regression,
        }


def resample_with_replacement(values):
    """Picks len(values) items from the list, allowing the same item to be
    picked more than once. This is what 'bootstrap resampling' means."""
    return [random.choice(values) for _ in values]


def detect_regression(baseline_scores, candidate_scores, n_bootstrap=2000):
    """Two average scores being different doesn't prove one version is
    actually worse - with a small test set, that difference could just be
    luck. This runs the comparison many times on randomly resampled data
    to build a range (a 95% confidence interval) for how big the true
    difference probably is. If that whole range is below zero, we can be
    confident the candidate is really worse, not just unlucky."""
    deltas = []
    for _ in range(n_bootstrap):
        baseline_sample = resample_with_replacement(baseline_scores)
        candidate_sample = resample_with_replacement(candidate_scores)
        delta = statistics.mean(candidate_sample) - statistics.mean(baseline_sample)
        deltas.append(delta)

    deltas.sort()
    # the values 2.5% and 97.5% of the way through the sorted list mark
    # the boundaries of the 95% confidence interval
    low_index = int(0.025 * n_bootstrap)
    high_index = int(0.975 * n_bootstrap)
    ci_low = deltas[low_index]
    ci_high = deltas[high_index]

    is_regression = ci_high < 0  # the whole interval is negative -> confidently worse

    return RegressionResult(
        baseline_mean=statistics.mean(baseline_scores),
        candidate_mean=statistics.mean(candidate_scores),
        ci_low=ci_low,
        ci_high=ci_high,
        is_regression=is_regression,
    )


def cohens_kappa(labels_a, labels_b):
    """Measures how much two graders agree, beyond what you'd expect from
    random chance. Used to check whether the LLM judge's scores actually
    match human judgment - a kappa below about 0.4 means don't trust the
    judge yet."""
    if len(labels_a) != len(labels_b):
        raise ValueError("label lists must be the same length")
    n = len(labels_a)
    if n == 0:
        raise ValueError("no labels given")

    # observed agreement: fraction of items where both graders gave the
    # exact same label
    agreements = 0
    for i in range(n):
        if labels_a[i] == labels_b[i]:
            agreements = agreements + 1
    observed_agreement = agreements / n

    # expected agreement by chance: for each possible label, how often
    # does rater A use it times how often rater B uses it
    all_labels = set(labels_a) | set(labels_b)
    expected_agreement = 0.0
    for label in all_labels:
        fraction_a = labels_a.count(label) / n
        fraction_b = labels_b.count(label) / n
        expected_agreement = expected_agreement + fraction_a * fraction_b

    if expected_agreement == 1.0:
        return 1.0

    return (observed_agreement - expected_agreement) / (1 - expected_agreement)
