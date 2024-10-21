import importlib
import json
from typing import Dict

import numpy as np

from evaluators.abstract_evaluator import AbstractEvaluator, Evaluation
# from game_tree.local_position_node import LocalPositionNode
from sgf_utils.game_node import GameNode

evaluator_classes = {
    'a0jax': 'evaluators.a0jax_evaluator.A0jaxEvaluator',
    'katago': 'evaluators.katago_evaluator.KatagoEvaluator',
    'a0kata': 'evaluators.a0kata_evaluator.A0KataEvaluator',
}


class EvaluationRegistry:
    def __init__(self, **initialized_evaluators):
        self.evaluation_registry: Dict[str, Evaluation] = {}
        self.evaluator_registry: Dict[str, AbstractEvaluator] = initialized_evaluators

    def get_evaluator(self, evaluator_name: str, **evaluator_kwargs) -> AbstractEvaluator:
        evaluator_key = self.get_key(evaluator_name, **evaluator_kwargs)
        if evaluator_key not in self.evaluator_registry:
            class_path = evaluator_classes[evaluator_name]
            self.evaluator_registry[evaluator_key] = self.instantiate_class(class_path, **evaluator_kwargs)
        return self.evaluator_registry[evaluator_key]

    @staticmethod
    def instantiate_class(class_path, **kwargs):
        parts = class_path.split('.')
        module_name = '.'.join(parts[:-1])
        class_name = parts[-1]
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        obj = cls(**kwargs)
        return obj

    @staticmethod
    def get_key(evaluator_name, **kwargs):
        kwargs = sorted(kwargs)
        return json.dumps({'name': evaluator_name, 'kwargs': kwargs})

    def __call__(self, position: GameNode, mask: np.ndarray, evaluator_name: str, **evaluator_kwargs):
        evaluator = self.get_evaluator(evaluator_name, **evaluator_kwargs)
        # print(evaluator_name, evaluator_kwargs)
        # print(evaluator, str(evaluator))
        evaluation_key = evaluator.get_key(position, mask)
        evaluation_key = json.dumps({'evaluator': evaluator_name, 'evaluation_key': evaluation_key})
        if evaluation_key not in self.evaluation_registry:
            self.evaluation_registry[evaluation_key] = Evaluation(evaluator, position, mask)
        cur_eval = self.evaluation_registry[evaluation_key]
        # evaluator.add_to_queue(cur_eval)
        return cur_eval
