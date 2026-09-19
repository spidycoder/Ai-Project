"""Judge calibration: run the LLM judge on a set of test cases that also
have human-assigned labels, then measure agreement. This must be run and
reported BEFORE the judge score is trusted for anything important - it's
the single number that separates a real eval platform from an unverified
'ask the LLM to grade itself' script."""
import json

from eval_platform.dataset import load_dataset
from eval_platform.stats import cohens_kappa


def load_human_labels(path):
    """Reads a JSONL file of {"test_case_id": ..., "human_score": 1-5}
    and returns a dict mapping test_case_id -> human_score."""
    labels = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            row = json.loads(line)
            labels[row["test_case_id"]] = int(row["human_score"])
    return labels


def calibrate_judge(dataset_path, human_labels_path, target, judge):
    """Returns {"kappa": ..., "n": ..., "per_case": [...]}. n is how many
    dataset rows actually had a matching human label; this raises an error
    if that's zero, which usually means the ids in your human-labels file
    don't match the dataset's test-case ids."""
    test_cases = load_dataset(dataset_path)
    human_labels = load_human_labels(human_labels_path)

    human_scores = []
    judge_scores = []
    per_case = []

    for test_case in test_cases:
        if test_case.id not in human_labels:
            continue

        response = target.invoke(test_case.input)
        result = judge.score(test_case, response)
        if result is None:
            continue  # judge unavailable (no API key) - skip rather than crash

        human_score = human_labels[test_case.id]
        judge_score = round(result.value)

        human_scores.append(human_score)
        judge_scores.append(judge_score)
        per_case.append({
            "test_case_id": test_case.id,
            "human_score": human_score,
            "judge_score": judge_score,
            "judge_reasoning": result.details,
        })

    if len(human_scores) == 0:
        raise ValueError(
            "No overlapping test_case_ids between dataset and human_labels file - "
            "check that ids match exactly."
        )

    kappa = cohens_kappa(human_scores, judge_scores)
    return {"kappa": kappa, "n": len(human_scores), "per_case": per_case}
