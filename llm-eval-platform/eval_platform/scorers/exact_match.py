"""Deterministic ground-truth scorer. Always prefer this over LLM-as-judge
when a test case has a known-correct answer - it's free, instant, and has
zero judge-bias risk."""
import re

from eval_platform.scorers.base import Scorer
from eval_platform.types import ScoreResult


def normalize(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


class ExactMatchScorer(Scorer):
    name = "exact_match"

    def score(self, test_case, response):
        if not test_case.expected_output:
            return None

        expected = normalize(test_case.expected_output)
        actual = normalize(response.output)

        # substring match, not full equality - RAG answers legitimately
        # include extra hedging/framing text around the core fact.
        is_hit = expected in actual

        return ScoreResult(
            scorer_name=self.name,
            value=1.0 if is_hit else 0.0,
            details="expected substring " + ("found" if is_hit else "NOT found") + " in output",
        )
