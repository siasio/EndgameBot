import copy
import json
from typing import Union, List

import numpy as np

from evaluators.a0jax_evaluator import A0jaxEvaluator
from evaluators.abstract_evaluator import AbstractEvaluator, Evaluation, pack_single_decorator
from evaluators.katago_evaluator import KatagoEvaluator
from sgf_utils.game_node import GameNode


class A0KataEvaluator(AbstractEvaluator):
    def __init__(self, a0_ckpt, a0_batch_size, kg_batch_size, **kata_kwargs):
        super().__init__()
        self.a0_evaluator = A0jaxEvaluator(a0_ckpt, a0_batch_size)
        self.kata_evaluator = KatagoEvaluator(kg_batch_size, **kata_kwargs)

    def set_moves_for_color(self, evaluation: Union[List[Evaluation], Evaluation], color: str):
        self.set_all(evaluation)

    def set_black_next_prob(self, evaluation: Union[List[Evaluation], Evaluation]):
        self.set_all(evaluation)

    def set_no_move_prob(self, evaluation: Union[List[Evaluation], Evaluation]):
        self.set_all(evaluation)

    def set_ownership(self, evaluation: Union[List[Evaluation], Evaluation]):
        self.set_all(evaluation)

    def set_score(self, evaluation: Union[List[Evaluation], Evaluation]):
        self.set_all(evaluation)

    def set_predicted_mask(self, evaluation: Union[List[Evaluation], Evaluation]):
        self.set_all(evaluation)

    @staticmethod
    def quantize(arr: np.ndarray):
        return round(arr / 10, 1) * 10

    def get_key(self, position: GameNode, mask: np.ndarray) -> str:
        key = {
            'stacked_pos': hash(stack_pos(position).data.tobytes()),
            'mask': hash(mask.data.tobytes())  # We should comment it out, but before we need to see how to update scores etc. for different masks
        }
        return json.dumps(key)

    @pack_single_decorator
    def set_all(self, evaluation: List[Evaluation]):
        kata_evals = copy.deepcopy(evaluation)
        a0_evals = copy.deepcopy(evaluation)
        self.kata_evaluator.set_all(kata_evals)
        self.a0_evaluator.set_all(a0_evals)
        for ev, kata_ev, a0_ev in zip(evaluation, kata_evals, a0_evals):
            ev.black_moves = a0_ev.black_moves
            ev.white_moves = a0_ev.white_moves
            ev.black_prob = a0_ev.black_prob
            ev.no_move_prob = max(kata_ev.no_move_prob, a0_ev.no_move_prob)
            ev.ownership = kata_ev.ownership
            ev.score = kata_ev.score
        self.inference_count = self.a0_evaluator.inference_count# + self.kata_evaluator.inference_count

    def add_to_queue(self, evaluation: Union[List['Evaluation'], 'Evaluation']):
        self.kata_evaluator.add_to_queue(evaluation)
        # if isinstance(evaluation, Evaluation):
        #     self.a0_evaluator.eval_queue.append(evaluation)
        # elif isinstance(evaluation, (list, tuple)):
        #     self.a0_evaluator.eval_queue.extend(evaluation)
        # else:
        #     raise ValueError("Expected Evaluation or a list of Evaluations")

    def shutdown(self):
        self.kata_evaluator.shutdown()
        self.a0_evaluator.shutdown()
        super().shutdown()


def get_position(node: GameNode):
    arr = np.zeros(node.board_size)
    for s in node.stones:
        if s.player == "W":
            arr[s.coords[0]][s.coords[1]] = - 1
        else:
            arr[s.coords[0]][s.coords[1]] = 1
    return arr


def stack_pos(node: GameNode):
    x, y = node.board_size
    previous_positions = []
    current_player = None
    while True:
        previous_positions.append(get_position(node))
        if len(previous_positions) >= 8 or node.parent is None or node.player == current_player:
            break
        current_player = node.player
        node = node.parent
    pad = 8 - len(previous_positions)
    stacked_pos = np.ones((9, x, y))
    for i in range(pad):
        stacked_pos[i] = previous_positions[-1]
    for i, pos in enumerate(previous_positions[::-1]):
        stacked_pos[i + pad] = pos
    stacked_pos = np.moveaxis(stacked_pos, 0, -1)
    # stacked_pos *= opponent_sign
    return stacked_pos
