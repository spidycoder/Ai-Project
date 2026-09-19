"""Adapter for evaluating a real, already-deployed system (your RAG
Document Assistant, the LLM gateway, etc.) over HTTP. Point EVAL_TARGET_URL
at any endpoint that accepts {"input": "..."} and returns
{"output": "...", "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0} back
- edit the two marked lines below if your endpoint's shape is different."""
import os
import time

import requests

from eval_platform.targets.base import TargetSystem
from eval_platform.types import TargetResponse


class HttpTarget(TargetSystem):
    name = "http-target"

    def __init__(self, url=None, timeout_seconds=30):
        self.url = url or os.environ.get("EVAL_TARGET_URL")
        if not self.url:
            raise ValueError("HttpTarget needs a url (pass one, or set EVAL_TARGET_URL)")
        self.timeout_seconds = timeout_seconds

    def invoke(self, input_text):
        start_time = time.perf_counter()

        response = requests.post(
            self.url,
            json={"input": input_text},           # <- adjust to your API's request shape
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        body = response.json()                     # <- adjust to your API's response shape

        return TargetResponse(
            output=body.get("output", body.get("answer", "")),
            latency_ms=elapsed_ms,
            tokens_in=body.get("tokens_in", 0),
            tokens_out=body.get("tokens_out", 0),
            cost_usd=body.get("cost_usd", 0.0),
        )
