from typing import Callable

from ._confusion_matrix import ConfusionMatrixEvaluator


EVALUATOR_REGISTRY = {
    "ConfusionMatrixEvaluator": ConfusionMatrixEvaluator,
}


def build_evaluator(mode, class_labels, evaluator_configs) -> object:

    name = evaluator_configs.get("name")

    if name in EVALUATOR_REGISTRY:
        return EVALUATOR_REGISTRY[name](mode, class_labels, **evaluator_configs)
