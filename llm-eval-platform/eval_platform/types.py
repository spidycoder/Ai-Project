"""Plain Python classes used to pass data between the different parts of
the eval platform. These are just containers for related values — each
one is a class with an __init__ that stores its arguments as attributes,
nothing fancier than that."""


class TestCase:
    def __init__(self, id, input, category, difficulty="medium",
                 expected_output=None, rubric=None):
        self.id = id
        self.input = input
        self.category = category
        self.difficulty = difficulty
        self.expected_output = expected_output
        self.rubric = rubric


class TargetResponse:
    def __init__(self, output, latency_ms, tokens_in=0, tokens_out=0, cost_usd=0.0):
        self.output = output
        self.latency_ms = latency_ms
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out
        self.cost_usd = cost_usd


class ScoreResult:
    def __init__(self, scorer_name, value, max_value=1.0, details=""):
        self.scorer_name = scorer_name
        self.value = value            # raw score, e.g. 0-1 or 1-5
        self.max_value = max_value    # what the raw score is out of
        self.details = details


class TestCaseResult:
    def __init__(self, test_case_id, category, input, expected_output,
                 actual_output, latency_ms, tokens_in, tokens_out, cost_usd, scores):
        self.test_case_id = test_case_id
        self.category = category
        self.input = input
        self.expected_output = expected_output
        self.actual_output = actual_output
        self.latency_ms = latency_ms
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out
        self.cost_usd = cost_usd
        self.scores = scores   # a list of ScoreResult objects

    def composite_score(self):
        """Average of all scores, each normalized to a 0-1 scale first
        (so a 4/5 judge score and a 1/1 exact-match score count equally)."""
        if len(self.scores) == 0:
            return 0.0
        total = 0.0
        for s in self.scores:
            total = total + (s.value / s.max_value)
        return total / len(self.scores)
