"""Rubric-based LLM-as-judge scorer, backed by Gemini (free-tier API key at
https://aistudio.google.com/apikey). This is the scorer that must be
calibrated against human labels (see eval_platform/calibration.py) before
its scores are trusted for anything important - never treat judge output
as ground truth on its own.

Gemini's free tier limits how many requests you can make per minute, so
this scorer waits a little before every call to stay under that limit
(see wait_for_my_turn below)."""
import json
import os
import re
import time

from eval_platform.scorers.base import Scorer
from eval_platform.types import ScoreResult

RUBRIC_TEMPLATE = (
    "You are grading an AI system's answer for faithfulness and correctness "
    "relative to the reference answer. Ignore style, verbosity, and tone.\n\n"
    "Score from 1-5:\n"
    "5 = fully correct and faithful to the reference\n"
    "4 = correct but missing minor detail\n"
    "3 = partially correct, some inaccuracy\n"
    "2 = mostly incorrect but topically relevant\n"
    "1 = incorrect, hallucinated, or off-topic\n\n"
    "Question: {question}\n"
    "Reference answer: {reference}\n"
    "System's answer: {actual}\n\n"
    "Respond with ONLY a JSON object: {{\"score\": <1-5 integer>, \"reasoning\": \"<one sentence>\"}}"
)


class LlmJudgeScorer(Scorer):
    name = "llm_judge"

    def __init__(self, model=None, api_key=None, requests_per_minute=None, max_retries=2):
        # gemini-2.0-flash was retired, and a pinned gemini-3.6-flash turned
        # out to have only a 20-requests/day free quota. The "-latest" lite
        # alias gets a higher free daily quota AND keeps tracking whichever
        # lite model Google currently recommends, instead of quietly
        # pointing at one they eventually retire.
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.requests_per_minute = requests_per_minute or float(os.environ.get("GEMINI_RPM", "5"))
        self.max_retries = max_retries
        self.time_of_last_call = 0.0
        self.client = None

    def get_client(self):
        if self.client is None:
            from google import genai  # only imported if the judge is actually used
            self.client = genai.Client(api_key=self.api_key)
        return self.client

    def wait_for_my_turn(self):
        """Sleeps just long enough that calls stay under requests_per_minute.
        For example at 5 requests/minute, this waits at least 12 seconds
        between calls."""
        seconds_between_calls = 60.0 / self.requests_per_minute
        seconds_since_last_call = time.time() - self.time_of_last_call
        seconds_to_wait = seconds_between_calls - seconds_since_last_call
        if seconds_to_wait > 0:
            time.sleep(seconds_to_wait)
        self.time_of_last_call = time.time()

    def score(self, test_case, response):
        if not self.api_key:
            return None  # no key configured - skip this scorer instead of crashing

        reference = test_case.expected_output or test_case.rubric or "(no reference provided; judge general quality)"
        prompt = RUBRIC_TEMPLATE.format(
            question=test_case.input, reference=reference, actual=response.output
        )

        from google.genai import types

        client = self.get_client()
        result = None
        last_error = None

        for attempt in range(self.max_retries + 1):
            self.wait_for_my_turn()
            try:
                result = client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(max_output_tokens=200, temperature=0),
                )
                break
            except Exception as error:
                last_error = error
                is_rate_limit_error = "429" in str(error) or "RESOURCE_EXHAUSTED" in str(error)
                is_last_attempt = attempt == self.max_retries
                if not is_rate_limit_error or is_last_attempt:
                    raise
                time.sleep(5)  # brief pause before retrying a rate-limit error

        if result is None:
            raise last_error

        text = result.text or "{}"
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return ScoreResult(self.name, 0.0, max_value=5.0,
                                details="judge returned unparseable output: " + text[:200])

        try:
            parsed = json.loads(match.group(0))
            score_value = float(parsed["score"])
            reasoning = parsed.get("reasoning", "")
        except (json.JSONDecodeError, KeyError, ValueError):
            return ScoreResult(self.name, 0.0, max_value=5.0,
                                details="judge JSON parse failed: " + text[:200])

        return ScoreResult(self.name, score_value, max_value=5.0, details=reasoning)
