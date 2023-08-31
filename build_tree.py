from local_pos_masks import AnalyzedPosition
from game_node import GameNode
from game import BaseGame
from sgf_parser import Move
import numpy as np
from typing import TypeVar, Generic
from enum import Enum
from utils import stack_last_state, add_new_state, add_new_stack_previous_state

import jax.numpy as jnp


class NextMovePlayer(Enum):
    none = "none"
    black = "black"
    white = "white"
    both = "both"


class LocalPositionNode(GameNode):
    def __init__(self, parent=None, properties=None, move=None, player=None):
        super().__init__(parent=parent, properties=properties, move=move, player=None)
        self.__a0pos = None
        self.next_move_player: NextMovePlayer = None

    @property
    def a0pos(self) -> AnalyzedPosition:
        return self.__a0pos

    @a0pos.setter
    def a0pos(self, a0pos: AnalyzedPosition):
        self.__a0pos = a0pos

    def check_final(self):
        if self.a0pos.black_move_prob < 0.1 and self.a0pos.white_move_prob < 0.1:
            self.next_move_player = NextMovePlayer.none
        elif self.a0pos.black_move_prob > 0.9:
            self.next_move_player = NextMovePlayer.black
        elif self.a0pos.white_move_prob > 0.9:
            self.next_move_player = NextMovePlayer.white
        else:
            self.next_move_player = NextMovePlayer.both

    @property
    def rounded_ownership(self):
        if self.next_move_player == NextMovePlayer.none:
            local_ownership = self.a0pos.predicted_ownership * self.a0pos.local_mask
            undecided_intersections = abs(local_ownership) < 0.6
            # Some of the intersections might have ownership close to zero due to seki
            undecided_intersections = undecided_intersections * (abs(local_ownership) > 0.2)
            if np.sum(undecided_intersections) >= 1:
                print("Position seems to be finished but some intersections have uncertain evaluation:")
                print(local_ownership)
            return np.round(local_ownership).astype(int)
        else:
            return self.a0pos.predicted_ownership * self.a0pos.local_mask

    @property
    def score(self):
        return np.sum(self.rounded_ownership)


class PositionTree(BaseGame[LocalPositionNode]):
    def __init__(self, position_node: LocalPositionNode):
        super().__init__(position_node)

    @classmethod
    def from_a0pos(cls, a0pos: AnalyzedPosition):
        white_sgf_stones = []
        black_sgf_stones = []
        for i in range(a0pos.size_x):
            for j in range(a0pos.size_y):
                color = a0pos.stones[i][j]
                if color == 1:
                    black_sgf_stones.append(Move(coords=(i, j), player='B').sgf([19, 19]))
                elif color == -1:
                    white_sgf_stones.append(Move(coords=(i, j), player='W').sgf([19, 19]))

        position_node: LocalPositionNode = LocalPositionNode(
            properties={'AW': white_sgf_stones, 'AB': black_sgf_stones, 'SZ': f'{str(a0pos.size_x)}:{str(a0pos.size_y)}'}
        )
        position_node.a0pos = a0pos
        return cls(position_node)

    def build_tree(self):
        local_mask = self.current_node.a0pos.local_mask if self.current_node.a0pos.fixed_mask else None
        self.current_node.a0pos.analyze_pos(local_mask)
        self.current_node.check_final()

    def play(self, move: Move, ignore_ko: bool = False):
        super().play(move=move, ignore_ko=ignore_ko)
        self.current_node.a0pos = AnalyzedPosition()
        self.update_a0pos_state()

    def update_a0pos_state(self):
        self.current_node.a0pos.stones = self.get_position()
        if self.current_node.parent is None:
            self.current_node.a0pos.stacked_pos = stack_last_state(jnp.array(self.current_node.a0pos.stones))
        else:
            previous_stacked_pos = self.current_node.parent.a0pos.stacked_pos
            color_to_play = - self.current_node.player_sign(self.current_node.player)
            if self.current_node.parent.player == self.current_node.player:
                self.current_node.a0pos.stacked_pos = add_new_stack_previous_state(previous_stacked_pos,
                                                                                   jnp.array(self.current_node.a0pos.stones),
                                                                                   color_to_play)
            else:
                self.current_node.a0pos.stacked_pos = add_new_state(previous_stacked_pos,
                                                                    jnp.array(self.current_node.a0pos.stones),
                                                                    color_to_play)

    def get_position(self):
        arr = np.zeros(self.board_size)
        for s in self.stones:
            if s.player == "W":
                arr[s.coords[0]][s.coords[1]] = - 1
            else:
                arr[s.coords[0]][s.coords[1]] = 1
        return arr