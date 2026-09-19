"""A self-contained mock RAG target so the whole pipeline runs end-to-end
with zero external dependencies. It does real (tiny) keyword-overlap
retrieval over a fixed knowledge base and deliberately gives a wrong
answer some percentage of the time, so the eval platform has real
mistakes to catch. Replace this with HttpTarget pointed at your actual
RAG assistant once it's ready."""
import hashlib
import random
import time

from eval_platform.targets.base import TargetSystem
from eval_platform.types import TargetResponse

KNOWLEDGE_BASE = {
    "refund policy": "Refunds are issued within 7 business days to the original payment method.",
    "shipping time": "Standard shipping takes 3-5 business days; express takes 1-2 business days.",
    "password reset": "Users can reset their password via the 'Forgot Password' link on the login page.",
    "api rate limit": "The public API allows 100 requests per minute per API key.",
    "data retention": "User data is retained for 90 days after account deletion, then permanently purged.",
}


def find_best_match(query):
    """Very simple retrieval: count how many words the query shares with
    each knowledge-base key, and return whichever key shares the most."""
    query_words = set(query.lower().split())
    best_key = None
    best_score = 0.0
    for key in KNOWLEDGE_BASE:
        key_words = set(key.split())
        overlap = len(query_words & key_words)
        score = overlap / len(key_words)
        if score > best_score:
            best_key = key
            best_score = score
    return best_key, best_score


class MockRagTarget(TargetSystem):
    name = "mock-rag-v1"

    def __init__(self, error_rate=0.15):
        # error_rate: fraction of answers deliberately made wrong, to
        # simulate a RAG system that sometimes retrieves the wrong passage.
        self.error_rate = error_rate

    def invoke(self, input_text):
        start_time = time.perf_counter()
        key, score = find_best_match(input_text)

        # Turn the input text into a number between 0 and 1 that's always
        # the same for the same input (so re-running the same dataset gives
        # the same results, instead of a different random answer each time).
        digest = hashlib.sha256(input_text.encode()).hexdigest()
        pseudo_random_value = int(digest, 16) % 1000 / 1000
        should_give_wrong_answer = pseudo_random_value < self.error_rate

        if key is None or score == 0.0:
            output = "I don't have information about that in the knowledge base."
        elif should_give_wrong_answer:
            output = "I'm not certain, but I believe this is handled automatically by the system."
        else:
            output = KNOWLEDGE_BASE[key]

        elapsed_ms = (time.perf_counter() - start_time) * 1000 + random.uniform(50, 250)
        tokens_in = max(len(input_text.split()), 1)
        tokens_out = max(len(output.split()), 1)

        return TargetResponse(
            output=output,
            latency_ms=elapsed_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=(tokens_in + tokens_out) * 0.000002,
        )
