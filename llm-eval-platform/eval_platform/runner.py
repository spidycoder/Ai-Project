"""Runs every test case in the dataset against a target system, scores
each one, and saves the results. This goes through the test cases one at
a time (not in parallel) - simpler to read and debug, and plenty fast for
a dataset of a few dozen test cases."""
import sys

from eval_platform.types import TestCaseResult


def run_one_test_case(test_case, target, scorers):
    response = target.invoke(test_case.input)

    scores = []
    for scorer in scorers:
        try:
            result = scorer.score(test_case, response)
        except Exception as error:
            # One scorer failing (e.g. the judge API being down) must not
            # throw away the other scorers' results for this test case.
            # We don't count it as a score of 0 either - a broken scorer is
            # not evidence the answer was bad, so it's just skipped, and
            # printed so a persistently broken scorer doesn't go unnoticed.
            print("[eval_platform] scorer '" + scorer.name + "' failed on test case '"
                  + test_case.id + "': " + str(error), file=sys.stderr)
            continue
        if result is not None:
            scores.append(result)

    return TestCaseResult(
        test_case_id=test_case.id,
        category=test_case.category,
        input=test_case.input,
        expected_output=test_case.expected_output,
        actual_output=response.output,
        latency_ms=response.latency_ms,
        tokens_in=response.tokens_in,
        tokens_out=response.tokens_out,
        cost_usd=response.cost_usd,
        scores=scores,
    )


def run_eval(test_cases, target, scorers, store, dataset_path, label=None, git_commit=None):
    run_id = store.start_run(target.name, dataset_path, git_commit=git_commit, label=label)

    all_results = []
    for test_case in test_cases:
        result = run_one_test_case(test_case, target, scorers)
        store.save_result(run_id, result)
        all_results.append(result)

    # group results by category so we can report a score per category,
    # not just one blended number
    scores_by_category = {}
    for result in all_results:
        category = result.category
        if category not in scores_by_category:
            scores_by_category[category] = []
        scores_by_category[category].append(result.composite_score())

    category_scores = {}
    for category, score_list in scores_by_category.items():
        category_scores[category] = sum(score_list) / len(score_list)

    all_scores = [result.composite_score() for result in all_results]
    overall_score = sum(all_scores) / len(all_scores) if len(all_scores) > 0 else 0.0

    store.finalize_run(run_id, overall_score, category_scores)

    return {
        "run_id": run_id,
        "n_cases": len(all_results),
        "composite_score": overall_score,
        "category_scores": category_scores,
    }
