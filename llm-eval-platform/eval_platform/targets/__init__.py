from eval_platform.targets.base import TargetSystem
from eval_platform.targets.mock_rag import MockRagTarget
from eval_platform.targets.http_target import HttpTarget

TARGET_TYPES = {
    "mock": MockRagTarget,
    "http": HttpTarget,
}


def build_target(name, **kwargs):
    if name not in TARGET_TYPES:
        raise ValueError("Unknown target '" + name + "'. Available: " + str(list(TARGET_TYPES.keys())))
    target_class = TARGET_TYPES[name]
    return target_class(**kwargs)
