from eval_platform.dataset import load_dataset
from eval_platform.runner import run_eval
from eval_platform.scorers import ExactMatchScorer, TextSimilarityScorer
from eval_platform.storage import ResultStore
from eval_platform.targets.mock_rag import MockRagTarget


def test_end_to_end_run(tmp_path):
    dataset_path = "datasets/golden_set.jsonl"
    test_cases = load_dataset(dataset_path)
    target = MockRagTarget(error_rate=0.0)  # deterministic, no injected errors, for a stable assertion
    scorers = [ExactMatchScorer(), TextSimilarityScorer()]
    store = ResultStore(tmp_path / "test.db")

    summary = run_eval(test_cases, target, scorers, store, dataset_path=dataset_path, label="unit-test")

    assert summary["n_cases"] == len(test_cases)
    assert 0.0 <= summary["composite_score"] <= 1.0
    assert "factual_accuracy" in summary["category_scores"]

    scores = store.get_run_scores(summary["run_id"])
    assert len(scores) == len(test_cases)


def test_run_persists_and_lists(tmp_path):
    dataset_path = "datasets/golden_set.jsonl"
    test_cases = load_dataset(dataset_path)
    target = MockRagTarget(error_rate=0.0)
    store = ResultStore(tmp_path / "test2.db")

    run_eval(test_cases, target, [ExactMatchScorer()], store, dataset_path=dataset_path, label="run-a")
    run_eval(test_cases, target, [ExactMatchScorer()], store, dataset_path=dataset_path, label="run-b")

    runs = store.list_runs()
    assert len(runs) == 2
