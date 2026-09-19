from eval_platform.scorers.exact_match import ExactMatchScorer
from eval_platform.scorers.text_similarity import TextSimilarityScorer
from eval_platform.types import TargetResponse, TestCase


def _tc(expected=None):
    return TestCase(id="t1", input="q", category="c", expected_output=expected)


def test_exact_match_hit():
    scorer = ExactMatchScorer()
    result = scorer.score(_tc("7 business days"), TargetResponse(output="Refunds take 7 business days.", latency_ms=1))
    assert result.value == 1.0


def test_exact_match_miss():
    scorer = ExactMatchScorer()
    result = scorer.score(_tc("7 business days"), TargetResponse(output="I don't know.", latency_ms=1))
    assert result.value == 0.0


def test_exact_match_none_when_no_expected():
    scorer = ExactMatchScorer()
    result = scorer.score(_tc(None), TargetResponse(output="anything", latency_ms=1))
    assert result is None


def test_text_similarity_identical():
    scorer = TextSimilarityScorer()
    result = scorer.score(_tc("the quick brown fox"), TargetResponse(output="the quick brown fox", latency_ms=1))
    assert result.value > 0.99


def test_text_similarity_unrelated():
    scorer = TextSimilarityScorer()
    result = scorer.score(_tc("the quick brown fox"), TargetResponse(output="completely different topic entirely", latency_ms=1))
    assert result.value < 0.3
