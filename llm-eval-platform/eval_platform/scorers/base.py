"""Every scorer is a class with a score(test_case, response) method that
returns a ScoreResult, or None if that scorer doesn't apply to this
particular test case (for example, exact-match has nothing to say when
there's no expected_output to compare against)."""


class Scorer(object):
    name = "unnamed-scorer"

    def score(self, test_case, response):
        raise NotImplementedError("Subclasses must implement score()")
