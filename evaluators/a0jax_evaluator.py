import json
import os.path
import pickle
from typing import List, Union

import jax.numpy as jnp
import numpy as np
from jax._src.nn.functions import softmax

from cgt_engine import L, R
from evaluators.abstract_evaluator import AbstractEvaluator, Evaluation, pack_single_decorator
from policies.resnet_policy import ResnetPolicyValueNet128, TransferResnet
from sgf_utils.game_node import GameNode

ownership_strict_threshold = 10


color_to_pl = {"B": L, "W": R}
pl_to_color = {L: "B", R: "W"}


class A0jaxEvaluator(AbstractEvaluator):
    def __init__(self, model_ckpt, a0_batch_size=16):
        super().__init__()
        self.agent = self.load_agent(os.path.join('models', model_ckpt))
        self.batch_size = a0_batch_size
        self.done_queue = []

    @staticmethod
    def load_agent(ckpt_path):
        backbone = ResnetPolicyValueNet128(input_dims=(9, 9, 9), num_actions=82)
        agent = TransferResnet(backbone)
        agent = agent.eval()
        with open(ckpt_path, "rb") as f:
            loaded_agent = pickle.load(f)
        loaded_agent = loaded_agent.get("agent", loaded_agent)
        agent = agent.load_state_dict(loaded_agent)
        return agent

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
            'mask': hash(mask.data.tobytes())
        }
        return json.dumps(key)

    @pack_single_decorator
    def set_all(self, evaluation: List[Evaluation]):
        # evaluation = [ev for ev in evaluation if ev not in self.done_queue]
        # if len(evaluation) == 0:
        #     self.inference_count += 1
        #     return
        # self.add_to_queue(evaluation)
        # different_evals = [ev for ev in self.eval_queue if ev not in evaluation]
        # num_to_grab = max(self.batch_size - len(evaluation), 0)  # In fact, we should also check if evaluation is not longer than batch_size
        # to_run_inference = different_evals[:num_to_grab]
        # evaluation.extend(to_run_inference)
        # print(f'Will inference batch of {len(evaluation)}')

        states = jnp.stack([stack_pos(ev.position) for ev in evaluation])
        board_shape = states.shape[1:3]
        board_size = np.product(board_shape)
        tensor_shape = [len(evaluation), *board_shape]
        opponent_signs = jnp.array([color_to_pl[ev.position.player].opp.sign for ev in evaluation])
        states = states * opponent_signs[:, None, None, None]
        masks = jnp.stack([ev.mask for ev in evaluation])
        board_masks = None
        action_logits, ownership_maps = self.agent((states, masks, board_masks), batched=True)
        ownership_maps = np.array(ownership_maps * opponent_signs[:, None, None])
        undecided_ownerships = jnp.sum((abs(ownership_maps) * 100 < (100 - ownership_strict_threshold)) * masks, axis=(1, 2)) > 0
        scores = jnp.sum(ownership_maps * masks, axis=(1, 2))
        # it doesn't serve its purpose when we have a seki, but in other cases it can stop the analysis when territories are decided


        action_logits = softmax(action_logits, axis=-1)
        black_probs = jnp.sum(action_logits[:, board_size:2 * board_size]) / jnp.sum(action_logits[:, :2 * board_size], axis=-1)
        no_move_probs = action_logits[:, -1]
        black_next_moves = action_logits[:, board_size:2 * board_size].reshape(tensor_shape) * masks
        black_next_moves /= jnp.sum(black_next_moves, axis=(1, 2), keepdims=True)
        white_next_moves = action_logits[:, :board_size].reshape(tensor_shape) * masks
        white_next_moves /= jnp.sum(white_next_moves, axis=(1, 2), keepdims=True)
        for black, no_move, black_moves, white_moves, ownership, score, undecided, opp, ev in zip(black_probs, no_move_probs, black_next_moves, white_next_moves, ownership_maps, scores, undecided_ownerships, opponent_signs, evaluation):
            ev.black_moves = black_moves if opp != -1 else white_moves
            ev.white_moves = white_moves if opp != -1 else black_moves
            ev.black_prob = black if opp != -1 else 1 - black
            ev.no_move_prob = max(no_move, 1 - undecided)
            ev.ownership = ownership
            ev.score = self.quantize(score)  # To treat 0.91 prediction as 1 (sure Black's territory) and 0.1 like 0 (seki)

        # self.eval_queue = [ev for ev in self.eval_queue if ev not in evaluation]
        # self.done_queue.extend(evaluation)
        # print(len(self.done_queue), len(self.eval_queue))

    def reset(self):
        self.done_queue = []
        super().reset()


def stack_pos(node: GameNode):
    x, y = node.board_size
    previous_positions = []
    current_player = None
    while True:
        previous_positions.append(node.stones)
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

