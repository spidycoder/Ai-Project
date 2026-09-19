"""Command-line entry point.

    python -m eval_platform.cli run --dataset datasets/golden_set.jsonl --target mock --label baseline
    python -m eval_platform.cli compare --baseline <run_id> --candidate <run_id>
    python -m eval_platform.cli calibrate --dataset datasets/golden_set.jsonl --human-labels datasets/human_labels.jsonl
    python -m eval_platform.cli list
"""
import argparse
import json
import subprocess
import sys

from dotenv import load_dotenv

from eval_platform.calibration import calibrate_judge
from eval_platform.dataset import load_dataset
from eval_platform.runner import run_eval
from eval_platform.scorers import ExactMatchScorer, LlmJudgeScorer, TextSimilarityScorer
from eval_platform.stats import detect_regression
from eval_platform.storage import ResultStore
from eval_platform.targets import build_target

load_dotenv()


def get_git_commit():
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        )
        return output.decode().strip()
    except Exception:
        return None


def cmd_run(args):
    test_cases = load_dataset(args.dataset)

    if args.target == "mock":
        target = build_target(args.target, error_rate=args.mock_error_rate)
    else:
        target = build_target(args.target)

    scorers = [ExactMatchScorer(), TextSimilarityScorer()]
    if not args.no_judge:
        scorers.append(LlmJudgeScorer())

    store = ResultStore(args.db)
    summary = run_eval(
        test_cases, target, scorers, store,
        dataset_path=args.dataset, label=args.label, git_commit=get_git_commit(),
    )
    print(json.dumps(summary, indent=2))


def cmd_compare(args):
    store = ResultStore(args.db)
    baseline_scores = store.get_run_scores(args.baseline)
    candidate_scores = store.get_run_scores(args.candidate)
    if len(baseline_scores) == 0 or len(candidate_scores) == 0:
        print("One of the runs has no results.", file=sys.stderr)
        sys.exit(2)

    result = detect_regression(baseline_scores, candidate_scores)
    print(json.dumps(result.as_dict(), indent=2))

    if result.is_regression:
        print("\nREGRESSION DETECTED: candidate is significantly worse than baseline "
              "(95% CI on delta: [" + str(round(result.ci_low, 4)) + ", "
              + str(round(result.ci_high, 4)) + "])", file=sys.stderr)
        sys.exit(1)
    print("\nNo statistically significant regression.")


def cmd_calibrate(args):
    target = build_target(args.target)
    judge = LlmJudgeScorer()
    result = calibrate_judge(args.dataset, args.human_labels, target, judge)
    print(json.dumps({"kappa": result["kappa"], "n": result["n"]}, indent=2))
    if result["kappa"] < 0.4:
        print("\nWARNING: kappa < 0.4 - judge agreement with humans is weak. "
              "Do not rely on this judge for anything important yet; revise the rubric prompt.",
              file=sys.stderr)


def cmd_list(args):
    store = ResultStore(args.db)
    for run in store.list_runs(limit=args.limit):
        print(json.dumps(run))


def main():
    parser = argparse.ArgumentParser(prog="eval_platform")
    parser.add_argument("--db", default="eval_results.db", help="SQLite results DB file path")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run the golden dataset against a target")
    p_run.add_argument("--dataset", required=True)
    p_run.add_argument("--target", default="mock", choices=["mock", "http"])
    p_run.add_argument("--label", default=None)
    p_run.add_argument("--no-judge", action="store_true", help="skip LLM-as-judge (no API key needed)")
    p_run.add_argument("--mock-error-rate", type=float, default=0.15,
                        help="only used with --target mock; fraction of answers to deliberately corrupt")
    p_run.set_defaults(func=cmd_run)

    p_cmp = sub.add_parser("compare", help="statistically compare two runs, fail (exit 1) on regression")
    p_cmp.add_argument("--baseline", required=True)
    p_cmp.add_argument("--candidate", required=True)
    p_cmp.set_defaults(func=cmd_compare)

    p_cal = sub.add_parser("calibrate", help="measure LLM-judge agreement with human labels")
    p_cal.add_argument("--dataset", required=True)
    p_cal.add_argument("--human-labels", required=True)
    p_cal.add_argument("--target", default="mock", choices=["mock", "http"])
    p_cal.set_defaults(func=cmd_calibrate)

    p_list = sub.add_parser("list", help="list recent runs")
    p_list.add_argument("--limit", type=int, default=20)
    p_list.set_defaults(func=cmd_list)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
