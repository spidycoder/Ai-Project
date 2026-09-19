"""Saves eval results to a local SQLite database file. SQLite needs no
separate server to install or run - it's just a single file on disk - which
keeps setup to zero for anyone running this project.

Every run is saved with a run_id so you can look back at any past run, and
every individual test case result is saved too, linked to its run.
"""
import json
import sqlite3
import time
import uuid

CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    target_name TEXT NOT NULL,
    dataset_path TEXT NOT NULL,
    git_commit TEXT,
    label TEXT,
    created_at REAL NOT NULL,
    composite_score REAL,
    category_scores_json TEXT
);

CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    test_case_id TEXT NOT NULL,
    category TEXT NOT NULL,
    input TEXT NOT NULL,
    expected_output TEXT,
    actual_output TEXT NOT NULL,
    latency_ms REAL NOT NULL,
    tokens_in INTEGER NOT NULL,
    tokens_out INTEGER NOT NULL,
    cost_usd REAL NOT NULL,
    composite_score REAL NOT NULL,
    scores_json TEXT NOT NULL
);
"""


class ResultStore(object):
    def __init__(self, db_path="eval_results.db"):
        self.db_path = db_path
        connection = self.connect()
        connection.executescript(CREATE_TABLES_SQL)
        connection.commit()
        connection.close()

    def connect(self):
        return sqlite3.connect(self.db_path)

    def start_run(self, target_name, dataset_path, git_commit=None, label=None):
        run_id = str(uuid.uuid4())[:8]
        connection = self.connect()
        connection.execute(
            "INSERT INTO runs (run_id, target_name, dataset_path, git_commit, label, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, target_name, dataset_path, git_commit, label, time.time()),
        )
        connection.commit()
        connection.close()
        return run_id

    def save_result(self, run_id, result):
        scores_as_dicts = []
        for s in result.scores:
            scores_as_dicts.append({
                "scorer_name": s.scorer_name,
                "value": s.value,
                "max_value": s.max_value,
                "details": s.details,
            })

        connection = self.connect()
        connection.execute(
            "INSERT INTO results (run_id, test_case_id, category, input, expected_output, "
            "actual_output, latency_ms, tokens_in, tokens_out, cost_usd, composite_score, scores_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id, result.test_case_id, result.category, result.input, result.expected_output,
                result.actual_output, result.latency_ms, result.tokens_in, result.tokens_out,
                result.cost_usd, result.composite_score(), json.dumps(scores_as_dicts),
            ),
        )
        connection.commit()
        connection.close()

    def finalize_run(self, run_id, composite_score, category_scores):
        connection = self.connect()
        connection.execute(
            "UPDATE runs SET composite_score = ?, category_scores_json = ? WHERE run_id = ?",
            (composite_score, json.dumps(category_scores), run_id),
        )
        connection.commit()
        connection.close()

    def get_run_scores(self, run_id):
        connection = self.connect()
        rows = connection.execute(
            "SELECT composite_score FROM results WHERE run_id = ? ORDER BY test_case_id", (run_id,)
        ).fetchall()
        connection.close()
        return [row[0] for row in rows]

    def get_run_summary(self, run_id):
        connection = self.connect()
        row = connection.execute(
            "SELECT run_id, target_name, dataset_path, label, created_at, composite_score, "
            "category_scores_json FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        connection.close()

        if row is None:
            raise ValueError("No such run: " + run_id)

        return {
            "run_id": row[0],
            "target_name": row[1],
            "dataset_path": row[2],
            "label": row[3],
            "created_at": row[4],
            "composite_score": row[5],
            "category_scores": json.loads(row[6]) if row[6] else {},
        }

    def list_runs(self, limit=20):
        connection = self.connect()
        rows = connection.execute(
            "SELECT run_id, target_name, label, created_at, composite_score FROM runs "
            "ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        connection.close()

        runs = []
        for row in rows:
            runs.append({
                "run_id": row[0], "target_name": row[1], "label": row[2],
                "created_at": row[3], "composite_score": row[4],
            })
        return runs
