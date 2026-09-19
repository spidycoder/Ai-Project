"""Cheap sanity-check scorer that compares word overlap between the
expected answer and the actual answer. This is intentionally NOT a real
embedding-similarity score - it's a near-zero-cost check that catches
wildly off-topic answers before spending judge-model tokens on them."""
import math
import re

from eval_platform.scorers.base import Scorer
from eval_platform.types import ScoreResult


def count_words(text):
    """Turns text into a dict of {word: how many times it appears}."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    counts = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    return counts


def cosine_similarity(counts_a, counts_b):
    if len(counts_a) == 0 or len(counts_b) == 0:
        return 0.0

    shared_words = set(counts_a.keys()) & set(counts_b.keys())
    dot_product = 0
    for word in shared_words:
        dot_product = dot_product + counts_a[word] * counts_b[word]

    length_a = math.sqrt(sum(count * count for count in counts_a.values()))
    length_b = math.sqrt(sum(count * count for count in counts_b.values()))

    if length_a == 0 or length_b == 0:
        return 0.0
    return dot_product / (length_a * length_b)


class TextSimilarityScorer(Scorer):
    name = "text_similarity"

    def score(self, test_case, response):
        if not test_case.expected_output:
            return None

        similarity = cosine_similarity(
            count_words(test_case.expected_output),
            count_words(response.output),
        )
        return ScoreResult(
            scorer_name=self.name,
            value=similarity,
            details="word-overlap similarity = " + str(round(similarity, 3)),
        )
