import copy
import math
import os
import re
import threading
from datetime import datetime
from typing import Dict, List, Optional, Union, TypeVar, Generic

# from kivy.clock import Clock

# from katrain.core.constants import (
#     OUTPUT_DEBUG,
#     OUTPUT_EXTRA_DEBUG,
#     OUTPUT_INFO,
#     PLAYER_AI,
#     PLAYER_HUMAN,
#     PROGRAM_NAME,
#     SGF_INTERNAL_COMMENTS_MARKER,
#     STATUS_ANALYSIS,
#     STATUS_ERROR,
#     STATUS_INFO,
#     STATUS_TEACHING,
#     PRIORITY_GAME_ANALYSIS,
#     PRIORITY_EXTRA_ANALYSIS,
#     PRIORITY_SWEEP,
#     PRIORITY_ALTERNATIVES,
#     PRIORITY_EQUALIZE,
#     PRIORITY_DEFAULT,
# )
# from katrain.core.game_node import GameNode
from sgf_parser import SGF, Move
from katrain_utils import var_to_grid
from game_node import GameNode

T = TypeVar('T', bound=GameNode)


class IllegalMoveException(Exception):
    pass


class KaTrainSGF(SGF):
    _NODE_CLASS = GameNode


class BaseGame(Generic[T]):
    """Represents a game of go, including an implementation of capture rules."""

    DEFAULT_PROPERTIES = {"GM": 1, "FF": 4}

    def __init__(
        self,
        # katrain,
        move_tree: T = None,
        # game_properties: Optional[Dict] = None,
        sgf_filename=None,
        # bypass_config=False,  # TODO: refactor?
    ):
        # self.katrain = katrain
        self._lock = threading.Lock()
        self.game_id = datetime.strftime(datetime.now(), "%Y-%m-%d %H %M %S")
        self.sgf_filename = sgf_filename

        self.insert_mode = False
        # self.external_game = False  # not generated by katrain at some point

        # if move_tree:
        self.root = move_tree
        # self.external_game = PROGRAM_NAME not in self.root.get_property("AP", "")
        handicap = int(self.root.handicap)
        num_starting_moves_black = 0
        node = self.root
        while node.children:
            node = node.children[0]
            if node.player == "B":
                num_starting_moves_black += 1
            else:
                break

        # if (
        #     handicap >= 2
        #     and not self.root.placements
        #     and not (num_starting_moves_black == handicap)
        #     and not (self.root.children and self.root.children[0].placements)
        # ):  # not really according to sgf, and not sure if still needed, last clause for fox
        #     self.root.place_handicap_stones(handicap)
        # else:
        #     default_properties = {**BaseGame.DEFAULT_PROPERTIES, "DT": self.game_id}
        #     if not bypass_config:
        #         default_properties.update(
        #             {
        #                 "SZ": katrain.config("game/size"),
        #                 "KM": katrain.config("game/komi"),
        #                 "RU": katrain.config("game/rules"),
        #             }
        #         )
        #     self.root = GameNode(
        #         properties={
        #             **default_properties,
        #             **(game_properties or {}),
        #         }
        #     )
        #     handicap = katrain.config("game/handicap")
        #     if not bypass_config and handicap:
        #         self.root.place_handicap_stones(handicap)

        # if not self.root.get_property("RU"):  # if rules missing in sgf, inherit current
        #     self.root.set_property("RU", katrain.config("game/rules"))

        self.current_node: T = self.root
        self.set_current_node(self.root)
        self.main_time_used = 0

        # restore shortcuts
        shortcut_id_to_node = {node.get_property("KTSID", None): node for node in self.root.nodes_in_tree}
        for node in self.root.nodes_in_tree:
            shortcut_id = node.get_property("KTSF", None)
            if shortcut_id and shortcut_id in shortcut_id_to_node:
                shortcut_id_to_node[shortcut_id].add_shortcut(node)

    # -- move tree functions --
    def _init_state(self):
        board_size_x, board_size_y = self.board_size
        self.board = [
            [-1 for _x in range(board_size_x)] for _y in range(board_size_y)
        ]  # type: List[List[int]]  #  board pos -> chain id
        self.chains = []  # type: List[List[Move]]  #   chain id -> chain
        self.prisoners = []  # type: List[Move]
        self.last_capture = []  # type: List[Move]

    def _calculate_groups(self):
        with self._lock:
            self._init_state()
            try:
                for node in self.current_node.nodes_from_root:
                    for m in node.move_with_placements:
                        self._validate_move_and_update_chains(
                            m, True
                        )  # ignore ko since we didn't know if it was forced
                    if node.clear_placements:  # handle AE by playing all moves left from empty board
                        clear_coords = {c.coords for c in node.clear_placements}
                        stones = [m for c in self.chains for m in c if m.coords not in clear_coords]
                        self._init_state()
                        for m in stones:
                            self._validate_move_and_update_chains(m, True)
            except IllegalMoveException as e:
                print(f'Current move was: {node.move}')
                stones = [m for c in self.chains for m in c]
                white_stones = [f'[{stone.sgf(self.board_size)}]' for stone in stones if stone.player == 'W']
                black_stones = [f'[{stone.sgf(self.board_size)}]' for stone in stones if stone.player == 'B']
                white = ''.join(white_stones)
                black = ''.join(black_stones)
                print(f'(;FF[4]GM[1]SZ[19]AW{white}AB{black})')
                print([(a.player, a.sgf(self.board_size)) for a in sum(self.chains, [])])
                raise Exception(f"Unexpected illegal move ({str(e)})")

    def _validate_move_and_update_chains(self, move: Move, ignore_ko: bool):
        board_size_x, board_size_y = self.board_size

        def neighbours(moves):
            return {
                self.board[m.coords[1] + dy][m.coords[0] + dx]
                for m in moves
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                if 0 <= m.coords[0] + dx < board_size_x and 0 <= m.coords[1] + dy < board_size_y
            }

        ko_or_snapback = len(self.last_capture) == 1 and self.last_capture[0] == move
        self.last_capture = []

        if move.is_pass:
            return

        if self.board[move.coords[1]][move.coords[0]] != -1:
            print(move.coords[1], move.coords[0])
            raise IllegalMoveException("Space occupied")

        # merge chains connected by this move, or create a new one
        nb_chains = list({c for c in neighbours([move]) if c >= 0 and self.chains[c][0].player == move.player})
        if nb_chains:
            this_chain = nb_chains[0]
            self.board = [[nb_chains[0] if sq in nb_chains else sq for sq in line] for line in self.board]
            for oc in nb_chains[1:]:
                self.chains[nb_chains[0]] += self.chains[oc]
                self.chains[oc] = []
            self.chains[nb_chains[0]].append(move)
        else:
            this_chain = len(self.chains)
            self.chains.append([move])
        self.board[move.coords[1]][move.coords[0]] = this_chain

        # check captures
        opp_nb_chains = {c for c in neighbours([move]) if c >= 0 and self.chains[c][0].player != move.player}
        for c in opp_nb_chains:
            if -1 not in neighbours(self.chains[c]):  # no liberties
                self.last_capture += self.chains[c]
                for om in self.chains[c]:
                    self.board[om.coords[1]][om.coords[0]] = -1
                self.chains[c] = []
        if ko_or_snapback and len(self.last_capture) == 1 and not ignore_ko:
            raise IllegalMoveException("Ko")
        self.prisoners += self.last_capture
        # SF: Always allow suicide
        # suicide: check rules and throw exception if needed
        if -1 not in neighbours(self.chains[this_chain]):
            # rules = self.rules
            # if len(self.chains[this_chain]) == 1:  # even in new zealand rules, single stone suicide is not allowed
            #     raise IllegalMoveException("Single stone suicide")
            # elif (isinstance(rules, str) and rules in ["tromp-taylor", "new zealand"]) or (
            #     isinstance(rules, dict) and rules.get("suicide", False)
            # ):
            if True:
                self.last_capture += self.chains[this_chain]
                for om in self.chains[this_chain]:
                    self.board[om.coords[1]][om.coords[0]] = -1
                self.chains[this_chain] = []
                self.prisoners += self.last_capture
            # else:  # suicide not allowed by rules
            #     raise IllegalMoveException("Suicide")

    # Play a Move from the current position, raise IllegalMoveException if invalid.
    def play(self, move: Move, ignore_ko: bool = False):
        board_size_x, board_size_y = self.board_size
        if not move.is_pass and not (0 <= move.coords[0] < board_size_x and 0 <= move.coords[1] < board_size_y):
            raise IllegalMoveException(f"Move {move} outside of board coordinates")
        try:
            # print(move, ignore_ko)
            self._validate_move_and_update_chains(move, ignore_ko)
        except IllegalMoveException as e:
            self._calculate_groups()
            # print(move, e)
            # print('Current move', move.coords)
            raise e
        with self._lock:
            played_node = self.current_node.play(move)
            self.current_node = played_node
        return played_node

    # Insert a list of moves from root, often just adding one.
    def sync_branch(self, moves: List[Move]):
        node = self.root
        with self._lock:
            for move in moves:
                node = node.play(move)
        return node

    def set_current_node(self, node: T):
        self.current_node: T = node
        self._calculate_groups()

    def undo(self, n_times=1, stop_on_mistake=None):
        break_on_branch = False
        cn = self.current_node  # avoid race conditions
        break_on_main_branch = False
        last_branching_node = cn
        if n_times == "branch":
            n_times = 9999
            break_on_branch = True
        elif n_times == "main-branch":
            n_times = 9999
            break_on_main_branch = True
        for move in range(n_times):
            if (
                stop_on_mistake is not None
                and cn.points_lost is not None
                and cn.points_lost >= stop_on_mistake
                # and self.katrain.players_info[cn.player].player_type != PLAYER_AI
            ):
                self.set_current_node(cn.parent)
                return
            previous_cn = cn
            if cn.shortcut_from:
                cn = cn.shortcut_from
            elif not cn.is_root:
                cn = cn.parent
            else:
                break  # root
            if break_on_branch and len(cn.children) > 1:
                break
            elif break_on_main_branch and cn.ordered_children[0] != previous_cn:  # implies > 1 child
                last_branching_node = cn
        if break_on_main_branch:
            cn = last_branching_node
        if cn is not self.current_node:
            self.set_current_node(cn)

    def redo(self, n_times=1, stop_on_mistake=None):
        cn = self.current_node  # avoid race conditions
        for move in range(n_times):
            if cn.children:
                child = cn.ordered_children[0]
                shortcut_to = [m for m, v in cn.shortcuts_to if child == v]  # are we about to go to a shortcut node?
                if shortcut_to:
                    child = shortcut_to[0]
                cn = child
            if (
                move > 0
                and stop_on_mistake is not None
                and cn.points_lost is not None
                and cn.points_lost >= stop_on_mistake
                # and self.katrain.players_info[cn.player].player_type != PLAYER_AI
            ):
                self.set_current_node(cn.parent)
                return
        if stop_on_mistake is None:
            self.set_current_node(cn)

    @property
    def komi(self):
        return self.root.komi

    @property
    def board_size(self):
        return self.root.board_size

    @property
    def stones(self):
        with self._lock:
            return sum(self.chains, [])

    @property
    def end_result(self):
        if self.current_node.end_state:
            return self.current_node.end_state
        # if self.current_node.parent and self.current_node.is_pass and self.current_node.parent.is_pass:
        #     return self.manual_score or i18n._("board-game-end")

    @property
    def prisoner_count(
        self,
    ) -> Dict:  # returns prisoners that are of a certain colour as {B: black stones captures, W: white stones captures}
        return {player: sum([m.player == player for m in self.prisoners]) for player in Move.PLAYERS}

    @property
    def rules(self):
        return self.root.ruleset  # KataGoEngine.get_rules(self.root.ruleset)

    @property
    def manual_score(self):
        rules = self.rules
        if (
            not self.current_node.ownership
            or str(rules).lower() not in ["jp", "japanese"]
            or not self.current_node.parent
            or not self.current_node.parent.ownership
        ):
            if not self.current_node.score:
                return None
            return self.current_node.format_score(round(2 * self.current_node.score) / 2) + "?"
        board_size_x, board_size_y = self.board_size
        mean_ownership = [(c + p) / 2 for c, p in zip(self.current_node.ownership, self.current_node.parent.ownership)]
        ownership_grid = var_to_grid(mean_ownership, (board_size_x, board_size_y))
        stones = {m.coords: m.player for m in self.stones}
        lo_threshold = 0.15
        hi_threshold = 0.85
        max_unknown = 10
        max_dame = 4 * (board_size_x + board_size_y)

        def japanese_score_square(square, owner):
            player = stones.get(square, None)
            if (
                (player == "B" and owner > hi_threshold)
                or (player == "W" and owner < -hi_threshold)
                or abs(owner) < lo_threshold
            ):
                return 0  # dame or own stones
            if player is None and abs(owner) >= hi_threshold:
                return round(owner)  # surrounded empty intersection
            if (player == "B" and owner < -hi_threshold) or (player == "W" and owner > hi_threshold):
                return 2 * round(owner)  # captured stone
            return math.nan  # unknown!

        scored_squares = [
            japanese_score_square((x, y), ownership_grid[y][x])
            for y in range(board_size_y)
            for x in range(board_size_x)
        ]
        num_sq = {t: sum([s == t for s in scored_squares]) for t in [-2, -1, 0, 1, 2]}
        num_unkn = sum(math.isnan(s) for s in scored_squares)
        prisoners = self.prisoner_count
        score = sum([t * n for t, n in num_sq.items()]) + prisoners["W"] - prisoners["B"] - self.komi
        # self.katrain.log(
        #     f"Manual Scoring: {num_sq} score by square with {num_unkn} unknown, {prisoners} captures, and {self.komi} komi -> score = {score}",
        #     OUTPUT_DEBUG,
        # )
        if num_unkn > max_unknown or (num_sq[0] - len(stones)) > max_dame:
            return None
        return self.current_node.format_score(score)

    def __repr__(self):
        return (
            "\n".join("".join(self.chains[c][0].player if c >= 0 else "-" for c in line) for line in self.board)
            + f"\ncaptures: {self.prisoner_count}"
        )

    def update_root_properties(self):
        # def player_name(player_info):
        #     if player_info.name and player_info.player_type == PLAYER_HUMAN:
        #         return player_info.name
        #     else:
        #         return f"{i18n._(player_info.player_type)} ({i18n._(player_info.player_subtype)}){SGF_INTERNAL_COMMENTS_MARKER}"

        root_properties = self.root.properties
        x_properties = {}
        # for bw in "BW":
        #     if not self.external_game:
        #         x_properties["P" + bw] = player_name(self.katrain.players_info[bw])
        #         player_info = self.katrain.players_info[bw]
        #         if player_info.player_type == PLAYER_AI:
        #             x_properties[bw + "R"] = rank_label(player_info.calculated_rank)
        if "+" in str(self.end_result):
            x_properties["RE"] = self.end_result
        self.root.properties = {**root_properties, **{k: [v] for k, v in x_properties.items()}}

    # def generate_filename(self):
    #     self.update_root_properties()
    #     player_names = {
    #         bw: re.sub(r"[\u200b\u3164'<>:\"/\\|?*]", "", self.root.get_property("P" + bw, bw)) for bw in "BW"
    #     }
    #     base_game_name = f"{PROGRAM_NAME}_{player_names['B']} vs {player_names['W']}"
    #     return f"{base_game_name} {self.game_id}.sgf"

    def write_sgf(self, filename: str, trainer_config: Optional[Dict] = None):
        # if trainer_config is None:
        #     trainer_config = self.katrain.config("trainer", {})
        # save_feedback = trainer_config.get("save_feedback", False)
        # eval_thresholds = trainer_config["eval_thresholds"]
        # save_analysis = trainer_config.get("save_analysis", False)
        # save_marks = trainer_config.get("save_marks", False)
        self.update_root_properties()
        # show_dots_for = {
        #     bw: trainer_config.get("eval_show_ai", True) or self.katrain.players_info[bw].human for bw in "BW"
        # }
        sgf = self.root.sgf(
            # save_comments_player=show_dots_for,
            # save_comments_class=save_feedback,
            # eval_thresholds=eval_thresholds,
            # save_analysis=save_analysis,
            # save_marks=save_marks,
        )
        self.sgf_filename = filename
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(sgf)
        return #  i18n._("sgf written").format(file_name=filename)
