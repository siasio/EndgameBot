import json
import time

import numpy as np

from base_katrain import KaTrainBase
from engine import KataGoEngine
from game import BaseGame
from game_node import GameNode
from sgf_parser import SGFNode, Move
from array_katrain_utils import add_stones_to_node, detect_move, number_to_color


config_path = r"config.json"

with open(config_path, 'r') as f:
    config = json.load(f)
katrain = KaTrainBase()


class KatagoWrapper:
    def __init__(self, logfile="katago_wrapper.log"):
        self.eng = KataGoEngine(katrain, config["engine"]) #, custom_logfile=logfile)

    def analyze_node(self, initial_node, mask):
        initial_node.analyze(self.eng, visits=1, include_policy=True)
        while True:
            if initial_node.analysis["ownership"] is not None:
                break
            time.sleep(.25)
        ownership = np.array(initial_node.analysis["ownership"], dtype=np.float32).reshape((19, 19))
        # rotate ownership array clockwise
        ownership = np.rot90(ownership, 3)
        policy_array = np.array(initial_node.analysis["policy"][:-1]).reshape((19, 19))
        policy_array = np.rot90(policy_array, 3)
        log_mask = (np.array(mask, dtype=np.float32) - 1) * 1_000_000
        policy_array = policy_array + log_mask
        policy = list(policy_array.flatten()) + [initial_node.analysis["policy"][-1]]
        return policy, ownership

    def __call__(self, game_state, batched):
        state, mask, board_mask = game_state
        color = int(state[0][0][-1])
        if color == 0:
            color = 1
        color = 1
        # for board_state in state[:-1]:
        size = np.sum(board_mask[0]), np.sum(board_mask[..., 0])
        initial_node = GameNode(properties={'SZ': '19:19'}) #.join(map(str, size))})
        state = np.moveaxis(np.array(state), -1, 0)
        add_stones_to_node(initial_node, state[0] * color)
        previous_board_state = state[0]
        for board_state in state[1:-1]:
            # if board_state != previous_board_state:
            move = detect_move(previous_board_state * color, board_state * color)
            if move is not None:
                coords, number = move
                player = number_to_color[int(number)]
                initial_node = initial_node.play(Move(coords=coords, player=player))
                previous_board_state = board_state

        initial_node._player = "W"
        black_policy, black_ownership = self.analyze_node(initial_node, mask)

        initial_node._player = "B"
        white_policy, white_ownership = self.analyze_node(initial_node, mask)

        policy = np.array(white_policy[:-1] + black_policy[:-1] + [(white_policy[-1] + black_policy[-1])/2])
        # flattened_mask = np.array(mask).flatten()
        # doubled_flattened_mask = np.concatenate([flattened_mask, flattened_mask, [1]])
        # policy = policy * doubled_flattened_mask
        ownership = (white_ownership + black_ownership) / 2

        return policy, ownership
