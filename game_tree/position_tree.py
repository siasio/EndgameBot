import traceback
from typing import Tuple, List

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, QObject
from tqdm import tqdm

from game_tree.local_position_node import LocalPositionNode, KoTree
from evaluators.abstract_evaluator import Evaluation
from sgf_utils import game
from sgf_utils.game import BaseGame
from evaluators.evaluation_registry import EvaluationRegistry
from sgf_utils.sgf_parser import Move


class TreeBuildingException(Exception):
    def __init__(self, message):
        self.message = message


class PositionTree(BaseGame[LocalPositionNode]):

    def __init__(self, position_node: LocalPositionNode, config, game_name=None, **initialized_evaluators):
        super().__init__(position_node)
        self.eval = EvaluationRegistry(**initialized_evaluators)
        self.config = config
        self.evaluator_name = config['evaluator']
        self.game_name = game_name
        self.iter_num = None
        self.inference_num = None
        self._arr = None
        self.built_tree = False
        if config['mask']['from'] == 'sgf':
            self.mask = self.get_mask(self.root)
        else:
            raise NotImplementedError("Only sgf masks are supported for now")

    def iter_nodes(self, node=None):
        if node is None:
            node = self.root
        yield node
        for child in node.children:
            yield from self.iter_nodes(child)

    def go_to_node(self, node: LocalPositionNode):
        moves_to_play = []
        # print("Going to node", node.move)
        while node.parent is not None:
            moves_to_play.append(node.move)
            node = node.parent
        while self.current_node.parent is not None:
            self.undo()
        for move in reversed(moves_to_play):
            self.play(move, ignore_ko=True)

    def select_node(self):
        while True:
            # print(self.current_node.move, end=' ')
            # print(f"Selecting {self.current_node.move}")
            if len(self.current_node.unfinished_children) == 0:
                # print("No unfinished children")
                # assert len(self.current_node.children) == 0, "Calculations are finished for all children but not for the parent!"
                # This can happen because of ko, so we skip the assertion
                assert not self.current_node.visited, f"Node {self.current_node} has been already visited!"
                self.current_node.visited = True
                # print()
                break
            # Pick child for which expanded_tree_depth is the lowest
            # print([child.move for child in self.current_node.unfinished_children])
            node = min(self.current_node.unfinished_children, key=lambda x: x.expanded_tree_depth)
            try:
                self.play(node.move, ignore_ko=False)
            except game.IllegalMoveException as e:
                # Check if IllegalMoveException was raised with text "Ko"
                # print(e)
                if "Ko" in str(e):
                    parent = self.current_node.parent
                    if parent.ko_tree is None or parent.ko_node is None:
                        assert parent.ko_tree is None and parent.ko_node is None
                        # By creating ko tree, we immediately set the ko_node attribute of the current node
                        parent.ko_tree = KoTree(parent)
                    # By adding a child to a ko node, we immediately set the ko_node and ko_tree attributes of the child
                    parent.ko_node.add_child(self.current_node)
                    self.current_node.children = [ch for ch in self.current_node.children if ch.move != node.move]
                else:
                    raise e

    def eval_for_node(self, node: LocalPositionNode = None) -> Evaluation:
        node = node or self.current_node
        return self.eval(node, self.mask, self.evaluator_name, **self.config['evaluator_kwargs'])

    def expand_node(self, evaluation: Evaluation):
        strategy = self.config['moves'].get('strategy', None)
        moves_to_play = []
        if self.config['moves']['from'] == 'evaluator':
            if evaluation.no_move_prob > self.config['moves']['no_move_threshold']:
                moves_to_play = []
            else:
                black_moves = evaluation.black_moves
                white_moves = evaluation.white_moves
                black_prob = evaluation.black_prob
                white_prob = evaluation.white_prob
                best_moves = [self.best_move(black_moves, 'B'), self.best_move(white_moves, 'W')]
                if strategy == 'best':
                    moves_to_play = best_moves
                elif strategy in ['threshold', 'best+threshold']:
                    threshold = self.config['moves']['threshold']
                    moves_to_play = self.good_moves(black_moves * black_prob, threshold, 'B') + self.good_moves(
                        white_moves * white_prob, threshold, 'W')
                    if strategy == 'best+threshold':
                        for move in best_moves:
                            if move not in moves_to_play:
                                moves_to_play = moves_to_play + [move]
                elif strategy == 'doublethreshold':
                    threshold = self.config['moves']['threshold']
                    second_threshold = self.config['moves']['second_threshold']
                    black_moves_to_play = self.good_moves(black_moves * black_prob, threshold, 'B')
                    white_moves_to_play = self.good_moves(white_moves * white_prob, threshold, 'W')
                    if black_moves_to_play and not white_moves_to_play:
                        white_moves_to_play = self.good_moves(white_moves * white_prob, second_threshold, 'W')
                    elif white_moves_to_play and not black_moves_to_play:
                        black_moves_to_play = self.good_moves(black_moves * black_prob, second_threshold, 'B')
                    moves_to_play = black_moves_to_play + white_moves_to_play
                elif strategy == 'estimateunlikely':
                    threshold = self.config['moves']['threshold']
                    sente_threshold = self.config['moves']['sente_threshold']

                    black_moves_to_play = self.good_moves(black_moves, threshold, 'B') if black_prob > sente_threshold else []  # was: * black_prob
                    white_moves_to_play = self.good_moves(white_moves, threshold, 'W') if white_prob > sente_threshold else []  # was: * white_prob
                    move_to_estimate = None
                    if not self.current_node == self.root:
                        if black_moves_to_play and not white_moves_to_play:
                            move_to_estimate = self.best_move(white_moves, 'W')
                        elif white_moves_to_play and not black_moves_to_play:
                            move_to_estimate = self.best_move(black_moves, 'B')
                    moves_to_play = black_moves_to_play + white_moves_to_play
                    if move_to_estimate is not None:
                        moves_to_play = moves_to_play + [move_to_estimate]
                else:
                    raise NotImplementedError
                if self.current_node == self.root:
                    for move in best_moves:
                        if move not in moves_to_play:
                            moves_to_play = moves_to_play + [move]
        for move in moves_to_play:
            try:
                self.play(move, ignore_ko=False)
                # new_evaluation = self.eval(self.current_node, self.mask, self.evaluator_name, **self.config['evaluator_kwargs'])
                # cur_evaluator = self.eval.get_evaluator(self.evaluator_name, **self.config['evaluator_kwargs'])
                # cur_evaluator.add_to_queue(new_evaluation)
                if strategy == 'estimateunlikely':
                    if move_to_estimate is not None and move == move_to_estimate:
                        estimated_eval: Evaluation = self.eval_for_node(self.current_node)
                        self.current_node.visited = True
                        estimated_eval.quantize_ownership = False
                        self.current_node.set_cgt_game(estimated_eval.score)
                self.undo()
            except game.IllegalMoveException as e:
                # Check if IllegalMoveException was raised with text "Ko"
                # print(e)
                if "Ko" in str(e):
                    parent = self.current_node.parent
                    if parent.ko_tree is None or parent.ko_node is None:
                        assert parent.ko_tree is None and parent.ko_node is None
                        # By creating ko tree, we immediately set the ko_node attribute of the current node
                        parent.ko_tree = KoTree(parent)
                    # By adding a child to a ko node, we immediately set the ko_node and ko_tree attributes of the child
                    parent.ko_node.add_child(self.current_node)
                else:
                    pass
                    # stones = self.get_position()
                    # print(self.position_as_string(stones))
                    # raise e
                continue

    def backup(self, evaluation: Evaluation):
        self.current_node.set_cgt_game(evaluation.score)
        while True:
            if self.current_node.parent is None:
                break
            self.undo()
            try:
                self.current_node.set_cgt_game()
            except:
                pass

    def build_tree(self, max_depth=None, delete_engine=True, reset_engine=False, verbose=True):
        def calculate(pbar=None):
            current_depth = 0
            iterations_on_this_depth = 0
            iter_num = 0
            while not self.current_node.finished_calculation:
                if pbar is not None:
                    # Show the current depth and the number of visited nodes on pbar
                    pbar.set_description(
                        f"[{self.game_name}] Depth: {str(current_depth).rjust(2)}, Iterations: {str(iterations_on_this_depth).rjust(3)}")
                iterations_on_this_depth += 1
                print("About to select node")
                self.select_node()
                evaluation: Evaluation = self.eval_for_node(self.current_node)
                try:
                    if max_depth is not None and current_depth >= max_depth:
                        pass
                        raise TreeBuildingException(f"Max depth exceeded")
                        # print(f"Reached max tree depth ({current_depth})")
                    else:
                        self.expand_node(evaluation)
                    self.backup(evaluation)
                except Exception as e:
                    raise TreeBuildingException(
                        f"Error while expanding node {self.current_node.move} at depth {current_depth}") from e
                assert self.current_node == self.root, "Current node is not root after backup"
                if self.current_node.expanded_tree_depth > current_depth:
                    current_depth = self.current_node.expanded_tree_depth
                    iterations_on_this_depth = 0
                if pbar is not None:
                    pbar.update(1)
                iter_num += 1
            return iter_num

        try:
            print("About to calculate")
            if verbose:
                with tqdm(position=0, leave=True) as pbar:
                    iter_num = calculate(pbar)
            else:
                iter_num = calculate(pbar=None)

            self.iter_num = iter_num
            self.inference_num = self.eval.get_evaluator(self.evaluator_name,
                                                         **self.config['evaluator_kwargs']).inference_count
        finally:
            if delete_engine:
                if verbose:
                    print("Shutting down engines")
                keys = list(self.eval.evaluator_registry.keys())
                for key in keys:
                    engine = self.eval.evaluator_registry.pop(key)
                    engine.shutdown()
            elif reset_engine:
                if verbose:
                    print("Resetting engines")
                keys = list(self.eval.evaluator_registry.keys())
                for key in keys:
                    engine = self.eval.evaluator_registry[key]
                    engine.reset()
            self.built_tree = True

    def reset_tree(self):
        self.current_node.children = []
        self.current_node.is_initialized = False
        self.current_node.finished_calculation = False
        self._arr = None

    def get_mask(self, node: LocalPositionNode):
        arr = np.zeros(self.board_size)
        for s in node.properties[self.config['mask']['marker']]:
            arr[Move.SGF_COORD.index(s[0])][self.board_size[1] - Move.SGF_COORD.index(s[1]) - 1] = 1
        return arr

    def play(self, move: Move, ignore_ko: bool = False):
        super().play(move, ignore_ko)
        self._arr = None

    def undo(self, n_times=1, stop_on_mistake=None):
        super().undo(n_times, stop_on_mistake)
        self._arr = None

    @property
    def arr(self):
        if self._arr is None:
            self._arr = np.zeros(self.board_size)
            for s in self.stones:
                if s.player == "W":
                    self._arr[s.coords[0]][s.coords[1]] = - 1
                else:
                    self._arr[s.coords[0]][s.coords[1]] = 1
        return self._arr

    @staticmethod
    def position_as_string(position: np.ndarray):
        intersection_to_str = {0: ".", 1: "X", -1: "O"}
        return "\n".join(["".join(list(map(lambda x: intersection_to_str[int(x)], row))) for row in position])

    @staticmethod
    def best_move(predictions, color) -> Move:
        coords = tuple(np.unravel_index(np.argmax(predictions, axis=None), predictions.shape))
        return Move(coords, player=color)

    @staticmethod
    def good_moves(predictions, threshold, color) -> List[Move]:
        coords = np.argwhere(predictions > threshold)
        return [Move(coord, player=color) for coord in coords]


class PyQtPositionTree(PositionTree, QThread):
    update_signal = pyqtSignal()
    error_signal = pyqtSignal(str)

    def __init__(self, position_node: LocalPositionNode, config, parent_widget, game_name=None,
                 **initialized_evaluators):
        position_node.game = self
        PositionTree.__init__(self, position_node, config, game_name=game_name, **initialized_evaluators)
        QObject.__init__(self, parent_widget)
        self.parent_widget = parent_widget
        # self.max_depth = max_depth

    def run(self):
        """Override run method of QThread to run calculations in the PyQt app"""
        try:
            self.build_tree(delete_engine=False, reset_engine=True, max_depth=10)
        except Exception as e:
            # Signalize the error to the parent widget
            self.error_signal.emit(traceback.format_exc())

        self.update_signal.emit()
        self.parent_widget.wait_for_paint_event()

    def go_to_node(self, node: LocalPositionNode):
        super().go_to_node(node)
        self.update_signal.emit()
        self.parent_widget.wait_for_paint_event()

    def undo(self, n_times=1, stop_on_mistake=None):
        self.parent_widget.last_number -= 1
        self.parent_widget.stone_numbers[self.current_node.move.coords[0]][self.current_node.move.coords[1]] = None
        super().undo(n_times=n_times, stop_on_mistake=stop_on_mistake)

    def play(self, move: Move, ignore_ko: bool = False):
        super().play(move, ignore_ko)

        self.parent_widget.last_number += 1
        self.parent_widget.stone_numbers[self.current_node.move.coords[0]][
            self.current_node.move.coords[1]] = self.parent_widget.last_number
