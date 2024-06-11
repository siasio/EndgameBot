import copy
import json
import time
from typing import List, Union

import numpy as np

from evaluators.abstract_evaluator import AbstractEvaluator, Evaluation, pack_single_decorator
from sgf_utils.base_katrain import KaTrainBase
from sgf_utils.engine import KataGoEngine
from sgf_utils.game_node import GameNode

ownership_strict_threshold = 10


class KatagoEvaluator(AbstractEvaluator):
    def __init__(self, kg_batch_size=16, **kwargs):
        super().__init__()
        katrain = KaTrainBase()
        self.eng = KataGoEngine(katrain, kwargs)
        self.batch_size = kg_batch_size
        self.nodes_dict = {}

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
    def quantize(arr: np.ndarray, quantize_ownership=True):
        if quantize_ownership:
            return np.round(arr / 10, 1) * 10
        else:
            return arr

    def get_key(self, position: GameNode, mask: np.ndarray) -> str:
        key = {
            'stacked_pos': hash(stack_pos(position).data.tobytes()),
            'mask': hash(mask.data.tobytes())  # We should comment it out, but before we need to see how to update scores etc. for different masks
        }
        return json.dumps(key)

    def add_to_queue(self, evaluation: Union[List['Evaluation'], 'Evaluation']):
        if isinstance(evaluation, Evaluation):
            # if evaluation in self.eval_queue:
            #     return
            # self.eval_queue.append(evaluation)
            node_w = copy.deepcopy(evaluation.position)
            node_w._player = 'W'
            node_b = copy.deepcopy(evaluation.position)
            node_b._player = 'B'
            ev_key = self.get_key(evaluation.position, evaluation.mask)
            self.nodes_dict[ev_key] = {
                'W': node_w,
                'B': node_b
            }

            for node in [node_w, node_b]:
                node.analyze(self.eng, visits=1, include_policy=True)
        elif isinstance(evaluation, (list, tuple)):
            not_done_evaluations = [ev for ev in evaluation if ev not in self.eval_queue]
            # self.eval_queue.extend(not_done_evaluations)

            nodes_w = [copy.deepcopy(ev.position) for ev in not_done_evaluations]
            for node in nodes_w:
                node._player = 'W'

            nodes_b = [copy.deepcopy(ev.position) for ev in not_done_evaluations]
            for node in nodes_b:
                node._player = 'B'

            for ev, node_w, node_b in zip(not_done_evaluations, nodes_w, nodes_b):
                ev_key = self.get_key(ev.position, ev.mask)
                self.nodes_dict[ev_key] = {
                    'W': node_w,
                    'B': node_b
                }
            for node in nodes_w + nodes_b:
                node.analyze(self.eng, visits=1, include_policy=True)

    @pack_single_decorator
    def set_all(self, evaluation: List[Evaluation]):
        self.add_to_queue(evaluation)
        keys = [self.get_key(ev.position, ev.mask) for ev in evaluation]
        nodes_w = [self.nodes_dict.get(key).get('W') for key in keys]
        nodes_b = [self.nodes_dict.get(key).get('B') for key in keys]
        masks = [ev.mask for ev in evaluation]

        while True:
            if all(node.analysis["ownership"] is not None for node in nodes_w + nodes_b):
                break
            time.sleep(.1)
            # print('.', end='')

        ownerships_w = [np.array(node.analysis["ownership"], dtype=np.float32).reshape((19, 19)) for node in nodes_w]
        ownerships_b = [np.array(node.analysis["ownership"], dtype=np.float32).reshape((19, 19)) for node in nodes_b]
        ownerships = [(ow_w + ow_b) / 2 for ow_w, ow_b in zip(ownerships_w, ownerships_b)]
        # rotate ownership array clockwise
        ownerships = [np.rot90(ownership, 3) for ownership in ownerships]
        policies_w = [np.array(node.analysis["policy"][:-1]).reshape((19, 19)) for node in nodes_w]
        policies_w = [np.rot90(policy_array, 3) + 2 * mask - 2 for policy_array, mask in zip(policies_w, masks)]
        policies_b = [np.array(node.analysis["policy"][:-1]).reshape((19, 19)) for node in nodes_b]
        policies_b = [np.rot90(policy_array, 3) + 2 * mask - 2 for policy_array, mask in zip(policies_b, masks)]

        undecided_ownerships = [np.sum((abs(ownership) * 100 < (100 - ownership_strict_threshold)) * masks) > 0 for ownership in ownerships]
        scores = [np.sum(self.quantize(ownership, ev.quantize_ownership) * mask) for ownership, mask, ev in zip(ownerships, masks, evaluation)]
        # it doesn't serve its purpose when we have a seki, but in other cases it can stop the analysis when territories are decided

        for black_moves, white_moves, ownership, score, undecided, ev in zip(policies_w, policies_b, ownerships, scores, undecided_ownerships, evaluation):
            ev.black_moves = black_moves
            ev.white_moves = white_moves
            ev.black_prob = .5
            ev.no_move_prob = 1 - undecided
            ev.ownership = ownership
            ev.score = score  # To treat 0.91 prediction as 1 (sure Black's territory) and 0.1 like 0 (seki)

        self.inference_count += 1
        keys = list(self.nodes_dict.keys())
        for key in keys:
            self.nodes_dict.pop(key)

    def shutdown(self):
        self.eng.shutdown(finish=True)
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

