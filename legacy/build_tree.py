# from PyQt5.QtCore import QThread, pyqtSignal, QObject
# from tqdm import tqdm
#
# from sgf_utils import game
# from local_pos_masks import AnalyzedPosition
# from sgf_utils.game_node import GameNode
# from sgf_utils.game import BaseGame
# from sgf_utils.sgf_parser import Move
# import numpy as np
# from typing import Tuple, List
# from enum import Enum
#
# from cgt_engine import G, Options, L, R, both_pl
# from jax_utils import stack_last_state, add_new_state, add_new_stack_previous_state
#
#
# class NextMovePlayer(Enum):
#     none = "none"
#     black = "black"
#     white = "white"
#     both = "both"
#
#
# color_to_pl = {"B": L, "W": R}
# pl_to_color = {L: "B", R: "W"}
#
#
# class LocalPositionNode(GameNode):
#     def __init__(self, parent=None, properties=None, move=None, player=None):
#         super().__init__(parent=parent, properties=properties, move=move, player=None)
#         self.__calculated_ownership = None
#         self.__next_move_player = None
#         self.__a0pos = None
#         self.is_initialized = False
#         self.__calculated_score = None
#         self.game: PositionTree = None
#         self.finished_calculation = False
#         self.cgt_game = None
#         self.ko_node: KoNode = None  # Wrapper over the node for ko calculations
#         self.ko_tree: KoTree = None  # There can be only one ko tree per node
#         self.visited = False
#
#     @property
#     def unfinished_children(self):
#         return [child for child in self.children if not child.finished_calculation]
#
#     @property
#     def expanded_tree_depth(self):
#         # Minimal depth of each of the node's children + 1
#         return min([child.expanded_tree_depth for child in self.unfinished_children]) + 1 if len(self.unfinished_children) > 0 else 0
#
#     @property
#     def a0pos(self) -> AnalyzedPosition:
#         return self.__a0pos
#
#     @a0pos.setter
#     def a0pos(self, a0pos: AnalyzedPosition):
#         self.__a0pos = a0pos
#
#     @property
#     def next_move_player(self):
#         if self.__next_move_player is None and self.is_initialized:
#             if self.a0pos.black_move_prob < 0.3 and self.a0pos.white_move_prob < 0.3:
#                 self.__next_move_player = NextMovePlayer.none
#             elif self.a0pos.black_move_prob < 0.1:
#                 self.__next_move_player = NextMovePlayer.white
#             elif self.a0pos.white_move_prob < 0.1:
#                 self.__next_move_player = NextMovePlayer.black
#             else:
#                 self.__next_move_player = NextMovePlayer.both
#         return self.__next_move_player
#
#     @next_move_player.setter
#     def next_move_player(self, value: NextMovePlayer):
#         self.__next_move_player = value
#
#     @property
#     def rounded_ownership(self):
#         if self.next_move_player == NextMovePlayer.none:
#             local_ownership = self.a0pos.predicted_ownership * self.a0pos.local_mask
#             undecided_intersections = abs(local_ownership) < 0.6
#             # Some of the intersections might have ownership close to zero due to seki
#             undecided_intersections = undecided_intersections * (abs(local_ownership) > 0.2)
#             if np.sum(undecided_intersections) >= 1:
#                 pass
#                 # print("Position seems to be finished but some intersections have uncertain evaluation:")
#                 # print((10 * local_ownership).astype(int))
#             local_ownership[abs(local_ownership) < 0.2] = 0
#             return np.sign(local_ownership).astype(int)
#         else:
#             return self.a0pos.predicted_ownership * self.a0pos.local_mask
#
#     @staticmethod
#     def best_move_coords(predictions) -> Tuple[int, int]:
#         return tuple(np.unravel_index(np.argmax(predictions, axis=None), predictions.shape))
#
#     def good_move_coords(self, predictions, threshold) -> List[Tuple[int, int]]:
#         coords = np.argwhere(predictions > threshold)
#         return [(coord[0], coord[1]) for coord in coords]
#
#     @property
#     def cur_score(self):
#         # if self.next_move_player == NextMovePlayer.none:
#         return np.sum(self.rounded_ownership)
#
#     @property
#     def calculated_score(self):
#         if self.__calculated_score is None:
#             self.__calculated_score = self.cur_score
#         return self.__calculated_score
#
#     @calculated_score.setter
#     def calculated_score(self, value):
#         self.__calculated_score = value
#
#     @property
#     def calculated_ownership(self):
#         if self.__calculated_ownership is None:
#             self.__calculated_ownership = self.rounded_ownership
#         return self.__calculated_ownership
#
#     @calculated_ownership.setter
#     def calculated_ownership(self, value):
#         self.__calculated_ownership = value
#
#     @property
#     def black_move(self):
#         if self.next_move_player == NextMovePlayer.none or self.next_move_player == NextMovePlayer.white:
#             return None
#         coords = self.best_move_coords(self.a0pos.predicted_black_next_moves)
#         return Move(coords, player="B")
#
#     @property
#     def white_move(self):
#         if self.next_move_player == NextMovePlayer.none or self.next_move_player == NextMovePlayer.black:
#             return None
#         coords = self.best_move_coords(self.a0pos.predicted_white_next_moves)
#         return Move(coords, player="W")
#
#     def black_moves(self, threshold):
#         if self.next_move_player == NextMovePlayer.none or self.next_move_player == NextMovePlayer.white:
#             return []
#         coords = self.good_move_coords(self.a0pos.predicted_black_next_moves, threshold)
#         return [Move(coord, player="B") for coord in coords]
#
#     def white_moves(self, threshold):
#         if self.next_move_player == NextMovePlayer.none or self.next_move_player == NextMovePlayer.black:
#             return []
#         coords = self.good_move_coords(self.a0pos.predicted_white_next_moves, threshold)
#         return [Move(coord, player="W") for coord in coords]
#
#     def cgt_info(self):
#         temp = float(self.cgt_game.temp)
#         mean = float(self.cgt_game.mean)
#         value = 2 * temp - 2
#         return f"{self.cgt_game}\nMean: {mean:.2f} Temp: {temp:.2f} Value: {value:.2f}"
#
#     def set_cgt_game(self):
#         self.finished_calculation = all(child.finished_calculation for child in self.children)
#
#         if len(self.children) == 0:
#             self.calculated_ownership = self.rounded_ownership
#             self.calculated_score = self.cur_score
#             self.cgt_game = G(float(self.cur_score))
#
#         elif self.finished_calculation:
#             if self.ko_node is None:
#                 assert all(child.cgt_game is not None for child in self.children), "Some children have no cgt game"
#                 phantom_value = color_to_pl[self.player].opp.worst
#                 left_options = Options(
#                     [child.cgt_game for child in self.children if child.player == 'B']
#                 )
#                 right_options = Options(
#                     [child.cgt_game for child in self.children if child.player == 'W']
#                 )
#                 if len(left_options) == 0:
#                     left_options = G(phantom_value)
#                 if len(right_options) == 0:
#                     right_options = G(phantom_value)
#                 # We know that there are some children because of the if condition.
#                 # But maybe there are only children of one color.
#                 # Be careful: the code below gives wrong results when the reason for the missing children
#                 # is a lack of any possible local moves for the player who has just played (e.g. seki).
#                 cgt_game = left_options | right_options
#                 self.cgt_game = cgt_game
#             elif self.ko_tree.built and not self.ko_tree.added_phantom_values:
#                 self.ko_tree.add_phantom_values()
#
#
#
# class KoNode:
#     def __init__(self, node: LocalPositionNode, is_root=False, parent: 'KoNode' = None):
#         self.node = node
#         self.is_root = is_root
#         if not self.is_root:
#             assert parent is not None
#             self.branch = node.player
#             self.children = {self.branch: []}
#             self.parent = parent
#             self.depth = parent.depth + 1
#         else:
#             self.branch = None
#             self.children = {'B': [], 'W': []}
#             self.parent = None
#             self.depth = 0
#         node.ko_node = self
#
#     def add_child(self, node: LocalPositionNode):
#         # Only nodes for which the recapture is possible are added to the ko tree
#         # It means that the ultimate nodes, i.e. moves finishing the ko, are not added
#         assert self.is_root or node.player == self.branch
#         child_node = KoNode(node, parent=self)
#         node.ko_node = child_node
#         node.ko_tree = self.node.ko_tree
#         self.children[node.player].append(child_node)
#
#
# class KoTree:
#     def __init__(self, ko_root: LocalPositionNode):
#         """
#         We initialize the ko chain from two nodes with open ko, assuming that node_child
#         is the first node in the position tree in which recapture was possible (but illegal).
#         This assumption might prove wrong in rare cases of multi-stage kos, in which the network
#         didn't return a ko recapture as a possible move in one of previous nodes.
#         Arguably, this is a network's fault and not a fault of this design.
#         It would be good to raise meaningful exceptions in such cases.
#         :param node_child: Node in which a player captured a ko, and recapture is illegal
#         :param node_parent: Node to which the position would revert, was the recapture legal in node_child
#         """
#
#         self.ko_root = KoNode(ko_root, is_root=True)
#         self.ko_root.ko_tree = self
#         self.added_phantom_values = False
#
#     def iter_nodes(self, node: KoNode = None):
#         if node is None:
#             node = self.ko_root
#         yield node
#         for branch in node.children.keys():
#             for child in node.children[branch]:
#                 yield from self.iter_nodes(child)
#
#     @property
#     def built(self):
#         return all(all(child.finished_calculation for child in ko_node.node.children if child.ko_tree != self) for ko_node in self.iter_nodes())
#
#     def add_phantom_values(self):
#         if self.added_phantom_values:
#             return
#         assert all(len(ko_node.children[pl]) <= 1 for ko_node in self.iter_nodes() for pl in ko_node.children.keys()), \
#             "Calculations for branching kos haven't been implemented yet"
#         stops = [self.ko_root, self.ko_root]
#         best_values = [pl.worst for pl in both_pl]
#         for pl in both_pl:
#             color = pl_to_color[pl]
#             while True:
#                 children = stops[pl].children
#                 if not children[color]:
#                     break
#                 stops[pl] = children[color][0]
#             penultimate_node = stops[pl].node
#             children_of_color = [c for c in penultimate_node.children if c.player == color and c.ko_tree != self]
#             if not children_of_color:
#                 print("Error")
#             assert children_of_color, f"No move finishing the ko for {color} in {self}"
#             for child in children_of_color:
#                 if child.cgt_game is None:
#                     print("Error")
#                 best_values[pl] = pl.better(best_values[pl], child.cgt_game.mean)
#         depths = [stops[pl].depth + 1 for pl in both_pl]
#         phantom_value = (best_values[L] - best_values[R]) / sum(depths)
#         for pl in both_pl:
#             # Here we add phantom options for the pl's opponent in the pl's branch of the ko tree
#
#             # Start from the penultimate ko node
#             ko_node = stops[pl]
#             while not ko_node.is_root:
#                 options_pair = [
#                     Options([c.cgt_game for c in ko_node.node.children if c.player == pl_to_color[p]])
#                     for p in both_pl
#                 ]
#                 # Stage from the perspective of the player who is going to recapture the ko
#                 stage_after_ko_recapture = depths[pl] - ko_node.depth + 1
#                 # Subtract the value for White recapturing the ko, add for Black
#                 score_after_ko_recapture = best_values[pl] - pl.sign * phantom_value * stage_after_ko_recapture
#                 options_pair[pl.opp] = Options(options_pair[pl.opp]) ^ G(score_after_ko_recapture)
#                 ko_node.node.cgt_game = options_pair[L] | options_pair[R]
#                 ko_node = ko_node.parent
#         options_pair = [
#             Options([c.cgt_game for c in self.ko_root.node.children if c.player == pl_to_color[p]])
#             for p in both_pl
#         ]
#         self.ko_root.node.cgt_game = options_pair[L] | options_pair[R]
#         self.added_phantom_values = True
#
#     def __str__(self):
#         return f"KoChain"
#
#     def __repr__(self):
#         return self.__str__()
#
#
#
# class PositionTree(BaseGame[LocalPositionNode], QThread):
#     update_signal = pyqtSignal()
#
#     def __init__(self, position_node: LocalPositionNode, parent_widget=None, game_name=None, expansion_strategy=None):
#         position_node.game = self
#         super(PositionTree, self).__init__(position_node)
#         QObject.__init__(self, parent_widget)
#         self.parent_widget = parent_widget
#         self.max_depth = 0
#
#         self.checked_nodes = []
#         self.score_stats = []
#         self.game_name = game_name
#
#     def iter_nodes(self, node=None):
#         if node is None:
#             node = self.root
#         yield node
#         for child in node.children:
#             yield from self.iter_nodes(child)
#
#     def load_agent(self, agent):
#         self.agent = agent
#
#     def print_debug_info(self, info, verbose=False):
#         if verbose:
#             print(info)
#
#     @classmethod
#     def from_a0pos(cls, a0pos: AnalyzedPosition, parent_widget=None, game_name=None):
#         white_sgf_stones = []
#         black_sgf_stones = []
#         for i in range(a0pos.size_x):
#             for j in range(a0pos.size_y):
#                 color = a0pos.stones[i][j]
#                 if color == 1:
#                     black_sgf_stones.append(Move(coords=(i, j), player='B').sgf([19, 19]))
#                 elif color == -1:
#                     white_sgf_stones.append(Move(coords=(i, j), player='W').sgf([19, 19]))
#
#         position_node: LocalPositionNode = LocalPositionNode(
#             properties={'AW': white_sgf_stones, 'AB': black_sgf_stones, 'SZ': f'{str(a0pos.size_x)}:{str(a0pos.size_y)}'}
#         )
#         position_node.a0pos = a0pos
#         return cls(position_node, parent_widget=parent_widget, game_name=game_name)
#
#     def go_to_node(self, node: LocalPositionNode):
#         moves_to_play = []
#         # print("Going to node", node.move)
#         while node.parent is not None:
#             moves_to_play.append(node.move)
#             node = node.parent
#         while self.current_node.parent is not None:
#             self.undo()
#         for move in reversed(moves_to_play):
#             self.play(move, ignore_ko=True)
#         if self.parent_widget is not None:
#             self.update_signal.emit()
#             self.parent_widget.wait_for_paint_event()
#
#     def select_node(self):
#         while True:
#             # print(f"Selecting {self.current_node.move}")
#             if len(self.current_node.unfinished_children) == 0:
#                 # print("No unfinished children")
#                 # assert len(self.current_node.children) == 0, "Calculations are finished for all children but not for the parent!"
#                 # This can happen because of ko, so we skip the assertion
#                 assert not self.current_node.visited, f"Node {self.current_node} has been already visited!"
#                 self.current_node.visited = True
#                 break
#             # Pick child for which expanded_tree_depth is the lowest
#             # print([child.move for child in self.current_node.unfinished_children])
#             node = min(self.current_node.unfinished_children, key=lambda x: x.expanded_tree_depth)
#             self.play(node.move, ignore_ko=True)
#
#     @staticmethod
#     def get_recommended_moves_per_node(node, threshold=0.1, use_multiple_moves=True):
#         if node.next_move_player == NextMovePlayer.none:
#             return []
#         elif not use_multiple_moves:
#             # Return top black and top white move
#             return [node.black_move, node.white_move]
#         else:
#             # Return all moves for which the prediction exceeds the threshold
#             return node.black_moves(threshold) + node.white_moves(threshold)
#
#     def expand_node(self, use_multiple_moves=False, multiple_threshold=0.1):
#         if self.current_node.next_move_player == NextMovePlayer.none:
#             self.current_node.finished_calculation = True
#             self.current_node.cgt_game = G(float(self.current_node.cur_score))
#         #     moves_to_play = []
#         # elif self.current_node.next_move_player == NextMovePlayer.black:
#         #     moves_to_play = [self.current_node.black_move]
#         # elif self.current_node.next_move_player == NextMovePlayer.white:
#         #     moves_to_play = [self.current_node.white_move]
#         # else:
#         #     moves_to_play = [self.current_node.black_move, self.current_node.white_move]
#         for move in self.get_recommended_moves_per_node(self.current_node, threshold=multiple_threshold, use_multiple_moves=use_multiple_moves):
#             if move is None:
#                 continue
#             try:
#                 self.play(move, ignore_ko=False)
#                 self.undo()
#             except game.IllegalMoveException as e:
#                 # Check if IllegalMoveException was raised with text "Ko"
#                 if "Ko" in str(e):
#                     parent = self.current_node.parent
#                     if parent.ko_tree is None or parent.ko_node is None:
#                         assert parent.ko_tree is None and parent.ko_node is None
#                         # By creating ko tree, we immediately set the ko_node attribute of the current node
#                         parent.ko_tree = KoTree(parent)
#                     # By adding a child to a ko node, we immediately set the ko_node and ko_tree attributes of the child
#                     parent.ko_node.add_child(self.current_node)
#                 else:
#                     stones = self.get_position()
#                     self.print_debug_info(self.position_as_string(stones))
#                     raise e
#                 continue
#
#     def backup(self):
#         while True:
#             self.current_node.set_cgt_game()
#             if self.current_node.parent is None:
#                 break
#             self.undo()
#
#     def run(self):
#         """Override run method of QThread to run calculations in the PyQt app"""
#         self.real_run()
#
#     def real_run(self, cgt_engine=True, multiple_threshold=0.1):
#         self.initialize_current_node(self.current_node.a0pos.local_mask)
#         current_depth = 0
#         iterations_on_this_depth = 0
#         with tqdm(position=0, leave=True) as pbar:
#             while not self.current_node.finished_calculation:
#                 # Show the current depth and the number of visited nodes on pbar
#                 pbar.set_description(f"[{self.game_name}] Depth: {str(current_depth).rjust(2)}, Iterations: {str(iterations_on_this_depth).rjust(3)}")
#                 iterations_on_this_depth += 1
#                 if self.parent_widget is not None:
#                     self.parent_widget.wait_for_paint_event()
#                 self.select_node()
#                 if self.parent_widget is not None:
#                     self.parent_widget.wait_for_paint_event()
#                 self.expand_node(use_multiple_moves=cgt_engine, multiple_threshold=multiple_threshold)
#                 self.backup()
#                 assert self.current_node == self.root, "Current node is not root after backup"
#                 if self.current_node.expanded_tree_depth > current_depth:
#                     current_depth = self.current_node.expanded_tree_depth
#                     self.checked_nodes.append(iterations_on_this_depth)
#                     self.score_stats.append(' '.join([f"{c.calculated_score:.2f}" for c in self.current_node.children]))
#                     iterations_on_this_depth = 0
#
#                 if self.current_node.expanded_tree_depth >= self.max_depth:
#                     self.print_debug_info(f"Reached max tree depth ({self.current_node.expanded_tree_depth})", verbose=True)
#                     self.current_node.finished_calculation = True
#                     # self.parent_widget.wait_for_paint_event()
#                 pbar.update(1)
#         if self.parent_widget is not None:
#             self.update_signal.emit()
#             self.parent_widget.wait_for_paint_event()
#
#     def play(self, move: Move, ignore_ko: bool = False, max_depth=8):
#         local_mask = self.current_node.a0pos.local_mask
#         super().play(move=move, ignore_ko=ignore_ko)
#         self.initialize_current_node(local_mask)
#         if self.parent_widget is not None:
#             self.parent_widget.last_number += 1
#             self.parent_widget.stone_numbers[self.current_node.move.coords[0]][self.current_node.move.coords[1]] = self.parent_widget.last_number
#
#     def undo(self, n_times=1, stop_on_mistake=None):
#         if self.parent_widget is not None:
#             self.parent_widget.last_number -= 1
#             self.parent_widget.stone_numbers[self.current_node.move.coords[0]][self.current_node.move.coords[1]] = None
#         super().undo(n_times=n_times, stop_on_mistake=stop_on_mistake)
#
#     def initialize_current_node(self, local_mask, verbose=False):
#         if self.current_node.is_initialized:
#             assert self.current_node.a0pos is not None, "Node is initialized but a0pos is None"
#             return
#         self.current_node.a0pos = AnalyzedPosition()
#         self.current_node.a0pos.local_mask = local_mask
#         self.current_node.game = self
#         self.update_a0pos_state(visualize_recent_positions=False)
#         self.current_node.a0pos.analyze_pos(local_mask, agent=self.agent)
#         if verbose:
#             print("STONES")
#             print(self.position_as_string(self.current_node.a0pos.stones))
#             # print(f"PREDICTED OWNERSHIP (color to play = {self.current_node.a0pos.color_to_play})")
#             # print(self.position_as_string(self.current_node.a0pos.stacked_pos[..., -2]))
#             # print(self.position_as_string(self.current_node.a0pos.predicted_ownership))
#             print("UNDECIDED")
#             und = (abs(self.current_node.a0pos.predicted_ownership) * 50 + 50 < (100 - 5 / 2)) * local_mask
#             print(self.position_as_string(und))
#             print(f"BLACK MOVE PROBABILITY: {self.current_node.a0pos.black_move_prob:.03f}")
#             print(f"WHITE MOVE PROBABILITY: {self.current_node.a0pos.white_move_prob:.03f}")
#             print(f"NO MOVE PROBABILITY: {self.current_node.a0pos.no_move_prob:.03f}")
#
#             # input("Press enter to continue")
#         self.current_node.is_initialized = True
#
#     def update_a0pos_state(self, visualize_recent_positions=False):
#         stones = self.get_position()
#         if visualize_recent_positions:
#             self.print_debug_info("Current position:")
#             self.print_debug_info(self.position_as_string(stones))
#         color_to_play = - self.current_node.player_sign(self.current_node.player)
#         self.current_node.a0pos.color_to_play = -1 if color_to_play == -1 else 1
#         if self.current_node.parent is None:
#             self.current_node.a0pos.stacked_pos = stack_last_state(np.array(stones))
#         else:
#             previous_stacked_pos = self.current_node.parent.a0pos.stacked_pos
#             # self.current_node.a0pos.stacked_pos = stack_last_state(np.array(stones))
#             if self.current_node.parent.player == self.current_node.player:
#                 self.current_node.a0pos.stacked_pos = add_new_stack_previous_state(previous_stacked_pos,
#                                                                                    np.array(stones),
#                                                                                    )
#             else:
#                 self.current_node.a0pos.stacked_pos = add_new_state(previous_stacked_pos,
#                                                                     np.array(stones),
#                                                                     )
#             # if visualize_recent_positions:
#             #     print("Stacked position:")
#             #     for i in range(8):
#             #         print(self.position_as_string(self.current_node.a0pos.stacked_pos[:, :, i]))
#             #         print("----")
#             # self.current_node.a0pos.stones *= color_to_play
#
#     def reset_tree(self):
#         # self.current_node = LocalPositionNode(properties=self.root.properties)
#         self.current_node.children = []
#         self.current_node.is_initialized = False
#         self.current_node.finished_calculation = False
#
#     def get_position(self):
#         arr = np.zeros(self.board_size)
#         for s in self.stones:
#             if s.player == "W":
#                 arr[s.coords[0]][s.coords[1]] = - 1
#             else:
#                 arr[s.coords[0]][s.coords[1]] = 1
#         return arr
#
#     @staticmethod
#     def position_as_string(position: np.ndarray):
#         def intersection_as_string(color) -> str:
#             color = int(color)
#             if color == 0:
#                 return "."
#             elif color == 1:
#                 return "X"
#             elif color == -1:
#                 return "O"
#         return "\n".join(["".join(list(map(lambda x: intersection_as_string(x), row))) for row in position])
