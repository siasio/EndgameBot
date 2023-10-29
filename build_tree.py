import time

from PyQt5.QtCore import QThread, pyqtSignal

import game
from local_pos_masks import AnalyzedPosition
from game_node import GameNode
from game import BaseGame
from sgf_parser import Move
import numpy as np
from typing import TypeVar, Generic, Tuple
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
        self.temperature = 0.0
        self.__calculated_ownership = None
        self.__next_move_player = None
        self.__a0pos = None
        self.is_initialized = False
        self.__calculated_score = None
        self.game: PositionTree = None
        self.finished_calculation = False
        self.total_ko_stage = 0
        self.current_ko_stage = 1
        self.continuation: LocalPositionNode = None
        self.answer: LocalPositionNode = None
        self.ko_starting_node_parent: LocalPositionNode = None
        self.ko_stopping_node_sibling: LocalPositionNode = None

    @property
    def ko_starting_node(self):
        if self.ko_starting_node_parent is None:
            return None
        return self.ko_starting_node_parent.continuation

    @property
    def ko_stopping_node(self):
        if self.ko_stopping_node_sibling is None or self.ko_stopping_node_sibling.parent is None:
            print("Ko stopping node sibling or its parent is None, this shouldn't ever happen!")
            return None
        if self.ko_stopping_node_sibling.parent.continuation is None or self.ko_stopping_node_sibling.parent.answer is None:
            print("Ko stopping node sibling parent continuation or answer is None")
            return None

        if self.ko_stopping_node_sibling.player == self.ko_stopping_node_sibling.parent.continuation.player:
            return self.ko_stopping_node_sibling.parent.answer
        else:
            return self.ko_stopping_node_sibling.parent.continuation

    @property
    def unfinished_children(self):
        return [child for child in self.children if not child.finished_calculation]

    @property
    def expanded_tree_depth(self):
        # Minimal depth of each of the node's children + 1
        return min([child.expanded_tree_depth for child in self.unfinished_children]) + 1 if len(self.unfinished_children) > 0 else 0

    @property
    def a0pos(self) -> AnalyzedPosition:
        return self.__a0pos

    @a0pos.setter
    def a0pos(self, a0pos: AnalyzedPosition):
        self.__a0pos = a0pos

    @property
    def next_move_player(self):
        if self.__next_move_player is None and self.is_initialized:
            if self.a0pos.black_move_prob < 0.3 and self.a0pos.white_move_prob < 0.3:
                self.__next_move_player = NextMovePlayer.none
            elif self.a0pos.black_move_prob < 0.1:
                self.__next_move_player = NextMovePlayer.white
            elif self.a0pos.white_move_prob < 0.1:
                self.__next_move_player = NextMovePlayer.black
            else:
                self.__next_move_player = NextMovePlayer.both
        return self.__next_move_player

    @next_move_player.setter
    def next_move_player(self, value: NextMovePlayer):
        self.__next_move_player = value

    @property
    def rounded_ownership(self):
        if self.next_move_player == NextMovePlayer.none:
            local_ownership = self.a0pos.predicted_ownership * self.a0pos.local_mask
            undecided_intersections = abs(local_ownership) < 0.6
            # Some of the intersections might have ownership close to zero due to seki
            undecided_intersections = undecided_intersections * (abs(local_ownership) > 0.2)
            if np.sum(undecided_intersections) >= 1:
                pass
                # print("Position seems to be finished but some intersections have uncertain evaluation:")
                # print((10 * local_ownership).astype(int))
            return np.round(local_ownership).astype(int)
        else:
            return self.a0pos.predicted_ownership * self.a0pos.local_mask

    @staticmethod
    def best_move_coords(predictions) -> Tuple[int, int]:
        return tuple(np.unravel_index(np.argmax(predictions, axis=None), predictions.shape))

    @property
    def cur_score(self):
        # if self.next_move_player == NextMovePlayer.none:
        return np.sum(self.rounded_ownership)

    @property
    def calculated_score(self):
        if self.__calculated_score is None:
            self.__calculated_score = self.cur_score
        return self.__calculated_score

    @calculated_score.setter
    def calculated_score(self, value):
        self.__calculated_score = value

    @property
    def calculated_ownership(self):
        if self.__calculated_ownership is None:
            self.__calculated_ownership = self.rounded_ownership
        return self.__calculated_ownership

    @calculated_ownership.setter
    def calculated_ownership(self, value):
        self.__calculated_ownership = value

    @property
    def black_move(self):
        if self.next_move_player == NextMovePlayer.none or self.next_move_player == NextMovePlayer.white:
            return None
        coords = self.best_move_coords(self.a0pos.predicted_black_next_moves)
        return Move(coords, player="B")

    @property
    def white_move(self):
        if self.next_move_player == NextMovePlayer.none or self.next_move_player == NextMovePlayer.black:
            return None
        coords = self.best_move_coords(self.a0pos.predicted_white_next_moves)
        return Move(coords, player="W")

    # @property
    # def calculated_score(self):
    #     if self.__calculated_score is None:
    #         self.calculate_score_and_ownership()
    #     return self.__calculated_score
    #
    # @calculated_score.setter
    # def calculated_score(self, value):
    #     self.__calculated_score = value
    #
    # @property
    # def calculated_ownership(self):
    #     if self.__calculated_ownership is None:
    #         self.calculate_score_and_ownership()
    #     return self.__calculated_ownership
    #
    # @calculated_ownership.setter
    # def calculated_ownership(self, value):
    #     self.__calculated_ownership = value
    #
    # def _set_current_score(self):
    #     self.calculated_score = self.cur_score
    #     self.calculated_ownership = self.rounded_ownership
    #
    # def _set_score_from_next(self, move):
    #     try:
    #         self.game.play(move)
    #     except game.IllegalMoveException as e:
    #         print(e)
    #         self._set_current_score()
    #         self.w_score = self.calculated_score
    #         self.b_score = self.calculated_score
    #         return
    #     self.game.current_node.a0pos.analyze_pos(self.a0pos.local_mask, agent=self.game.agent)
    #     self.calculated_score = self.game.current_node.calculated_score
    #     self.calculated_ownership = self.game.current_node.calculated_ownership
    #     self.game.undo()
    #
    # def _set_score_from_two_next(self, move_b, move_w):
    #     try:
    #         self.game.play(move_w)
    #     except game.IllegalMoveException as e:
    #         print(e)
    #         self._set_score_from_next(move_b)
    #         self.w_score = self.calculated_score
    #         self.b_score = self.calculated_score
    #         return
    #     print(self.game)
    #     print(f"Analyzing {move_w}, local mask: {np.sum(self.a0pos.local_mask)}")
    #     self.game.current_node.a0pos.analyze_pos(self.a0pos.local_mask, agent=self.game.agent)
    #     score_w = self.game.current_node.calculated_score
    #     ownership_w = self.game.current_node.calculated_ownership
    #     self.game.undo()
    #     try:
    #         self.game.play(move_b)
    #     except game.IllegalMoveException as e:
    #         print(e)
    #         self._set_score_from_next(move_w)
    #         self.w_score = self.calculated_score
    #         self.b_score = self.calculated_score
    #         return
    #     print(f"Analyzing {move_b}, local mask: {np.sum(self.a0pos.local_mask)}")
    #     self.game.current_node.a0pos.analyze_pos(self.a0pos.local_mask, agent=self.game.agent)
    #     score_b = self.game.current_node.calculated_score
    #     ownership_b = self.game.current_node.calculated_ownership
    #     self.game.undo()
    #     self.calculated_score = (score_w + score_b) / 2
    #     self.calculated_ownership = (ownership_w + ownership_b) / 2
    #     self.b_score = score_b
    #     self.w_score = score_w
    #     print("Calculated B score", self.b_score)
    #     print("Calculated W score", self.w_score)

    # def calculate_score_and_ownership(self):
    #     print(self.black_move, "-", self.a0pos.black_move_prob)
    #     print(self.white_move, "-", self.a0pos.white_move_prob)
    #     # print(self.a0pos.local_mask)
    #     if self.black_move is None and self.white_move is None:
    #         print("Variation end")
    #         self.next_move_player = NextMovePlayer.none
    #         self._set_current_score()
    #         self.w_score = self.calculated_score
    #         self.b_score = self.calculated_score
    #     elif self.black_move is None:
    #         self.next_move_player = NextMovePlayer.white
    #         self._set_score_from_next(self.white_move)
    #         self.w_score = self.calculated_score
    #         self.b_score = self.calculated_score
    #     elif self.white_move is None:
    #         self.next_move_player = NextMovePlayer.black
    #         self._set_score_from_next(self.black_move)
    #         self.w_score = self.calculated_score
    #         self.b_score = self.calculated_score
    #     else:
    #         self.next_move_player = NextMovePlayer.both
    #         self._set_score_from_two_next(self.black_move, self.white_move)

    def set_score_and_ownership(self):
        if len(self.children) == 0:
            # print("No children, setting ownership from current prediction")
            self.calculated_ownership = self.rounded_ownership
            self.calculated_score = self.cur_score
        else:
            # print("Setting ownership from children prediction")
            scores = []
            ownerships = []
            finished_calculation = True
            for child in self.children:
                if not child.finished_calculation:
                    finished_calculation = False
                scores.append(child.calculated_score)
                ownerships.append(child.calculated_ownership)
            # print("Scores", scores)
            self.finished_calculation = finished_calculation
            if len(scores) == 1 and self.total_ko_stage == 0:
                self.calculated_score = scores[0]
                self.calculated_ownership = ownerships[0]
                self.temperature = 0.0

            else:
                if self.total_ko_stage > 0:
                    print(f"Ko stage ({self.move}): {self.current_ko_stage}/{self.total_ko_stage + 2}")
                    scores = [self.ko_starting_node.calculated_score, self.ko_stopping_node.calculated_score]
                    ownerships = [self.ko_starting_node.calculated_ownership, self.ko_stopping_node.calculated_ownership]
                    factor = self.current_ko_stage / (self.total_ko_stage + 2)
                    print(f"Scores: {self.ko_starting_node.move} {scores[0]:.03f} - {self.ko_stopping_node.move} {scores[1]:.03f}, factor: {factor:.03f}")
                else:
                    factor = 1 / len(scores)
                    print(f"No ko ({self.move})")
                self.calculated_score = scores[1] * factor + scores[0] * (1 - factor)
                self.calculated_ownership = np.sum(ownerships[1], axis=0) * factor + np.sum(ownerships[0], axis=0) * (1 - factor)
                print(f"Calculated temperature: {self.temperature:.03f}")
                self.temperature = abs(scores[0] - scores[1]) * 2 / (self.total_ko_stage + 2)


class PositionTree(BaseGame[LocalPositionNode], QThread):
    update_signal = pyqtSignal()

    def __init__(self, position_node: LocalPositionNode, parent_widget=None):
        position_node.game = self
        # self.update_func = update_func
        super(PositionTree, self).__init__(position_node)
        QThread.__init__(self, parent_widget)
        self.parent_widget = parent_widget

    def load_agent(self, agent):
        self.agent = agent

    @classmethod
    def from_a0pos(cls, a0pos: AnalyzedPosition, parent_widget=None):
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
        return cls(position_node, parent_widget=parent_widget)

    def go_to_node(self, node: LocalPositionNode):
        moves_to_play = []
        # print("Going to node", node.move)
        while node.parent is not None:
            moves_to_play.append(node.move)
            node = node.parent
        while self.current_node.parent is not None:
            self.undo()
        for move in reversed(moves_to_play):
            self.play(move)
        self.update_signal.emit()
        self.parent_widget.wait_for_paint_event()

    def select_node(self):
        while True:
            # print(f"Selecting {self.current_node.move}")
            if len(self.current_node.unfinished_children) == 0:
                # print("No unfinished children")
                assert len(self.current_node.children) == 0, "Calculations are finished for all children but not for the parent!"
                return
            # Pick child for which expanded_tree_depth is the lowest
            # print([child.move for child in self.current_node.unfinished_children])
            node = min(self.current_node.unfinished_children, key=lambda x: x.expanded_tree_depth)
            self.play(node.move)

    def expand_node(self):
        if self.current_node.next_move_player == NextMovePlayer.none:
            self.current_node.finished_calculation = True
        #     moves_to_play = []
        # elif self.current_node.next_move_player == NextMovePlayer.black:
        #     moves_to_play = [self.current_node.black_move]
        # elif self.current_node.next_move_player == NextMovePlayer.white:
        #     moves_to_play = [self.current_node.white_move]
        # else:
        #     moves_to_play = [self.current_node.black_move, self.current_node.white_move]
        moves_to_play = [self.current_node.black_move, self.current_node.white_move]
        for move in moves_to_play:
            if move is None:
                continue
            try:
                self.play(move, ignore_ko=False)
                last_player = self.current_node.parent.player if self.current_node.parent is not None else "W"
                if self.current_node.player == last_player:
                    self.current_node.parent.continuation = self.current_node
                else:
                    self.current_node.parent.answer = self.current_node
                self.undo()
            except game.IllegalMoveException as e:
                print("Illegal move!!", e)
                # Check if IllegalMoveException was raised with text "Ko"
                if "Ko" in str(e):
                    parent_ko_stage = self.current_node.parent.total_ko_stage if self.current_node.parent is not None else 0
                    node = self.current_node
                    for i in range(parent_ko_stage):
                        try:
                            node = node.parent
                        except AttributeError:
                            raise AttributeError("Ko stage is higher than the tree depth! Perhaps, the problem is the ko starting in the first move?")
                        # if i == parent_ko_stage:
                    ko_stopping_node_sibling = node
                    # ko_stopping_node_color = node.move.opponent if node.move is not None else "B"
                    node = self.current_node
                    for i in range(parent_ko_stage + 1):
                        node.total_ko_stage = parent_ko_stage + 1
                        node.current_ko_stage = i + 1
                        node.ko_starting_node_parent = self.current_node
                        node.ko_stopping_node_sibling = ko_stopping_node_sibling
                        print(node.ko_starting_node_parent.move)
                        print(node.ko_stopping_node_sibling.move)
                        node = node.parent
                    print("Ko stage", self.current_node.total_ko_stage)
                else:
                    stones = self.get_position()
                    print(self.position_as_string(stones))
                    raise(e)
                continue
        # print('Children', ' '.join([str(child.move) for child in self.current_node.children]))

    def backup(self):
        while True:
            self.current_node.set_score_and_ownership()
            # print(f'{self.current_node.move} calculated score: {self.current_node.calculated_score:.03f}')
            if self.current_node.parent is None:
                break
            self.undo()

    def run(self, max_depth=8):
        # self.current_node.a0pos.analyze_pos(self.current_node.a0pos.local_mask, agent=self.agent)
        self.initialize_current_node(self.current_node.a0pos.local_mask)
        while not self.current_node.finished_calculation:
            self.parent_widget.wait_for_paint_event()
            self.select_node()
            # time.sleep(1)
            # self.update_signal.emit()
            # time.sleep(2)
            self.parent_widget.wait_for_paint_event()
            self.expand_node()
            self.backup()

            if self.current_node.expanded_tree_depth >= max_depth:
                print(f"Reached max tree depth ({self.current_node.expanded_tree_depth})")
                self.current_node.finished_calculation = True
                # self.parent_widget.wait_for_paint_event()
        self.update_signal.emit()
        self.parent_widget.wait_for_paint_event()

    def play(self, move: Move, ignore_ko: bool = False, max_depth=8):
        local_mask = self.current_node.a0pos.local_mask
        super().play(move=move, ignore_ko=ignore_ko)
        # self.current_node.id = self.parent_widget.last_number
        self.initialize_current_node(local_mask)
        self.parent_widget.last_number += 1
        self.parent_widget.stone_numbers[self.current_node.move.coords[0]][self.current_node.move.coords[1]] = self.parent_widget.last_number
        # print("Coords", self.current_node.move.coords)
        # self.update_func(to_print="PRINT THIS")
        # if self.goban is not None:
        #     print("UPDATING GOBAN")
        #     time.sleep(2)
        #     self.goban.update()
        # else:
        #     print("GOBAN IS NONE")

    def undo(self, n_times=1, stop_on_mistake=None):
        self.parent_widget.last_number -= 1
        self.parent_widget.stone_numbers[self.current_node.move.coords[0]][self.current_node.move.coords[1]] = None
        # print("Coords", self.current_node.move.coords)
        super().undo(n_times=n_times, stop_on_mistake=stop_on_mistake)

    def initialize_current_node(self, local_mask):
        if self.current_node.is_initialized:
            # print(f"Already initialized node {self.current_node.move}")
            assert self.current_node.a0pos is not None, "Node is initialized but a0pos is None"
            return
        # print(f"Initializing node {self.current_node.move}")
        self.current_node.a0pos = AnalyzedPosition()
        self.current_node.a0pos.local_mask = local_mask
        self.current_node.game = self
        self.update_a0pos_state(visualize_recent_positions=False)
        self.current_node.a0pos.analyze_pos(local_mask, agent=self.agent)
        self.current_node.is_initialized = True

    def update_a0pos_state(self, visualize_recent_positions=False):
        stones = self.get_position()
        if visualize_recent_positions:
            print("Current position:")
            print(self.position_as_string(stones))
        color_to_play = - self.current_node.player_sign(self.current_node.player)
        self.current_node.a0pos.color_to_play = -1 if color_to_play == -1 else 1
        if self.current_node.parent is None:
            self.current_node.a0pos.stacked_pos = stack_last_state(np.array(stones))
        else:
            previous_stacked_pos = self.current_node.parent.a0pos.stacked_pos
            if self.current_node.parent.player == self.current_node.player:
                self.current_node.a0pos.stacked_pos = add_new_stack_previous_state(previous_stacked_pos,
                                                                                   np.array(stones),
                                                                                   )
            else:
                self.current_node.a0pos.stacked_pos = add_new_state(previous_stacked_pos,
                                                                    np.array(stones),
                                                                    )
            # if visualize_recent_positions:
            #     print("Stacked position:")
            #     for i in range(8):
            #         print(self.position_as_string(self.current_node.a0pos.stacked_pos[:, :, i]))
            #         print("----")
            # self.current_node.a0pos.stones *= color_to_play

    def reset_tree(self):
        # self.current_node = LocalPositionNode(properties=self.root.properties)
        self.current_node.children = []
        self.current_node.is_initialized = False
        self.current_node.finished_calculation = False

    def get_position(self):
        arr = np.zeros(self.board_size)
        for s in self.stones:
            if s.player == "W":
                arr[s.coords[0]][s.coords[1]] = - 1
            else:
                arr[s.coords[0]][s.coords[1]] = 1
        return arr

    @staticmethod
    def position_as_string(position: np.ndarray):
        def intersection_as_string(color) -> str:
            color = int(color)
            if color == 0:
                return "."
            elif color == 1:
                return "X"
            elif color == -1:
                return "O"
        return "\n".join(["".join(list(map(lambda x: intersection_as_string(x), row))) for row in position])
