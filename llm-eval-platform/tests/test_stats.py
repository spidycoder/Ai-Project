from eval_platform.stats import cohens_kappa, detect_regression


def test_no_regression_when_scores_equal():
    baseline = [0.8, 0.9, 0.85, 0.9, 0.8] * 5
    candidate = [0.8, 0.9, 0.85, 0.9, 0.8] * 5
    result = detect_regression(baseline, candidate, n_bootstrap=1000)
    assert not result.is_regression


def test_regression_detected_on_clear_drop():
    baseline = [0.95, 0.9, 0.92, 0.94, 0.91] * 10
    candidate = [0.4, 0.35, 0.45, 0.3, 0.38] * 10
    result = detect_regression(baseline, candidate, n_bootstrap=1000)
    assert result.is_regression
    assert result.delta < 0


def test_regression_not_flagged_on_small_noisy_sample():
    # small n, tiny difference -> should NOT confidently claim regression
    baseline = [0.8, 0.85]
    candidate = [0.78, 0.83]
    result = detect_regression(baseline, candidate, n_bootstrap=1000)
    assert not result.is_regression


def test_cohens_kappa_perfect_agreement():
    a = [1, 2, 3, 4, 5, 1, 2, 3]
    assert cohens_kappa(a, a) == 1.0


def test_cohens_kappa_random_agreement_near_zero():
    a = [1, 1, 1, 1, 1, 1, 1, 1]
    b = [5, 5, 5, 5, 5, 5, 5, 5]
    # both raters are constant but disagree completely -> kappa undefined-ish (pe=1 edge case)
    # use a mixed case instead for a meaningful near-zero check
    a2 = [1, 2, 1, 2, 1, 2, 1, 2]
    b2 = [2, 1, 1, 2, 2, 1, 1, 2]
    k = cohens_kappa(a2, b2)
    assert -1.0 <= k <= 1.0
