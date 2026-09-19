"""Loads the golden dataset from a JSONL file (one JSON object per line)."""
import json

from eval_platform.types import TestCase


def load_dataset(path):
    test_cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "" or line.startswith("#"):
                continue
            row = json.loads(line)
            test_case = TestCase(
                id=row["id"],
                input=row["input"],
                category=row.get("category", "uncategorized"),
                difficulty=row.get("difficulty", "medium"),
                expected_output=row.get("expected_output"),
                rubric=row.get("rubric"),
            )
            test_cases.append(test_case)

    if len(test_cases) == 0:
        raise ValueError(path + " is empty")

    return test_cases
