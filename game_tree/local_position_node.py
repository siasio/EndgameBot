from cgt_engine import G, Options, L, R, both_pl
from evaluators.abstract_evaluator import Evaluation
from sgf_utils.game_node import GameNode
from sgf_utils.sgf_parser import SGF

color_to_pl = {"B": L, "W": R}
pl_to_color = {L: "B", R: "W"}


class LocalPositionNode(GameNode):
    def __init__(self, parent=None, properties=None, move=None, player=None):
        super().__init__(parent=parent, properties=properties, move=move, player=None)
        self.finished_calculation = False
        self.cgt_game = None
        self.ko_node: KoNode = None  # Wrapper over the node for ko calculations
        self.ko_tree: KoTree = None  # There can be only one ko tree per node
        self.visited = False

    @property
    def unfinished_children(self):
        return [child for child in self.children if not child.finished_calculation]

    @property
    def expanded_tree_depth(self):
        # Minimal depth of each of the node's children + 1
        return min([child.expanded_tree_depth for child in self.unfinished_children]) + 1 if len(self.unfinished_children) > 0 else 0

    def cgt_info(self):
        temp = float(self.cgt_game.temp)
        mean = float(self.cgt_game.mean)
        value = 2 * temp - 2
        return f"{self.cgt_game}\nMean: {mean:.2f} Temp: {temp:.2f} Value: {value:.2f}"

    def set_cgt_game(self, score=None):
        self.finished_calculation = all(child.finished_calculation for child in self.children)

        if len(self.children) == 0:
            # self.calculated_ownership = self.rounded_ownership
            # self.calculated_score = self.cur_score
            self.cgt_game = G(float(score))

        elif self.finished_calculation:
            if self.ko_node is None:
                assert all(child.cgt_game is not None for child in self.children), "Some children have no cgt game"
                phantom_value = color_to_pl[self.player].opp.worst
                left_options = Options(
                    [child.cgt_game for child in self.children if child.player == 'B']
                )
                right_options = Options(
                    [child.cgt_game for child in self.children if child.player == 'W']
                )
                if len(left_options) == 0:
                    left_options = G(phantom_value)
                if len(right_options) == 0:
                    right_options = G(phantom_value)
                # We know that there are some children because of the if condition.
                # But maybe there are only children of one color.
                # Be careful: the code below gives wrong results when the reason for the missing children
                # is a lack of any possible local moves for the player who has just played (e.g. seki).
                cgt_game = left_options | right_options
                self.cgt_game = cgt_game
            elif self.ko_tree.built and not self.ko_tree.added_phantom_values:
                self.ko_tree.add_phantom_values()


class KoNode:
    def __init__(self, node: LocalPositionNode, is_root=False, parent: 'KoNode' = None):
        self.node = node
        self.is_root = is_root
        if not self.is_root:
            assert parent is not None
            self.branch = node.player
            self.children = {self.branch: []}
            self.parent = parent
            self.depth = parent.depth + 1
        else:
            self.branch = None
            self.children = {'B': [], 'W': []}
            self.parent = None
            self.depth = 0
        node.ko_node = self

    def add_child(self, node: LocalPositionNode):
        # Only nodes for which the recapture is possible are added to the ko tree
        # It means that the ultimate nodes, i.e. moves finishing the ko, are not added
        assert self.is_root or node.player == self.branch
        child_node = KoNode(node, parent=self)
        node.ko_node = child_node
        node.ko_tree = self.node.ko_tree
        self.children[node.player].append(child_node)


class KoTree:
    def __init__(self, ko_root: LocalPositionNode):
        """
        We initialize the ko chain from two nodes with open ko, assuming that node_child
        is the first node in the position tree in which recapture was possible (but illegal).
        This assumption might prove wrong in rare cases of multi-stage kos, in which the network
        didn't return a ko recapture as a possible move in one of previous nodes.
        Arguably, this is a network's fault and not a fault of this design.
        It would be good to raise meaningful exceptions in such cases.
        :param node_child: Node in which a player captured a ko, and recapture is illegal
        :param node_parent: Node to which the position would revert, was the recapture legal in node_child
        """

        self.ko_root = KoNode(ko_root, is_root=True)
        self.ko_root.ko_tree = self
        self.added_phantom_values = False

    def iter_nodes(self, node: KoNode = None):
        if node is None:
            node = self.ko_root
        yield node
        for branch in node.children.keys():
            for child in node.children[branch]:
                yield from self.iter_nodes(child)

    @property
    def built(self):
        return all(all(child.finished_calculation for child in ko_node.node.children if child.ko_tree != self) for ko_node in self.iter_nodes())

    def add_phantom_values(self):
        if self.added_phantom_values:
            return
        assert all(len(ko_node.children[pl]) <= 1 for ko_node in self.iter_nodes() for pl in ko_node.children.keys()), \
            "Calculations for branching kos haven't been implemented yet"
        stops = [self.ko_root, self.ko_root]
        best_values = [pl.worst for pl in both_pl]
        for pl in both_pl:
            color = pl_to_color[pl]
            while True:
                children = stops[pl].children
                if not children[color]:
                    break
                stops[pl] = children[color][0]
            penultimate_node = stops[pl].node
            children_of_color = [c for c in penultimate_node.children if c.player == color and c.ko_tree != self]
            if not children_of_color:
                print("Error")
            assert children_of_color, f"No move finishing the ko for {color} in {self}"
            for child in children_of_color:
                if child.cgt_game is None:
                    print("Error")
                best_values[pl] = pl.better(best_values[pl], child.cgt_game.mean)
        depths = [stops[pl].depth + 1 for pl in both_pl]
        phantom_value = (best_values[L] - best_values[R]) / sum(depths)
        for pl in both_pl:
            # Here we add phantom options for the pl's opponent in the pl's branch of the ko tree

            # Start from the penultimate ko node
            ko_node = stops[pl]
            while not ko_node.is_root:
                options_pair = [
                    Options([c.cgt_game for c in ko_node.node.children if c.player == pl_to_color[p]])
                    for p in both_pl
                ]
                # Stage from the perspective of the player who is going to recapture the ko
                stage_after_ko_recapture = depths[pl] - ko_node.depth + 1
                # Subtract the value for White recapturing the ko, add for Black
                score_after_ko_recapture = best_values[pl] - pl.sign * phantom_value * stage_after_ko_recapture
                options_pair[pl.opp] = Options(options_pair[pl.opp]) ^ G(score_after_ko_recapture)
                ko_node.node.cgt_game = options_pair[L] | options_pair[R]
                ko_node = ko_node.parent
        options_pair = [
            Options([c.cgt_game for c in self.ko_root.node.children if c.player == pl_to_color[p]])
            for p in both_pl
        ]
        self.ko_root.node.cgt_game = options_pair[L] | options_pair[R]
        self.added_phantom_values = True

    def __str__(self):
        return f"KoChain"

    def __repr__(self):
        return self.__str__()


class LocalPositionSGF(SGF):
    _NODE_CLASS = LocalPositionNode
