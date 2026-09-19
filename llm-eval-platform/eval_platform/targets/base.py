"""The system-under-test interface. Every target (a mock, a real HTTP
endpoint, anything) is a class with an invoke(input_text) method that
returns a TargetResponse. This base class isn't required by Python to make
that work — it's just here so every target class documents the same shape
and a typo'd method name fails loudly instead of silently."""


class TargetSystem(object):
    name = "unnamed-target"

    def invoke(self, input_text):
        raise NotImplementedError("Subclasses must implement invoke()")
