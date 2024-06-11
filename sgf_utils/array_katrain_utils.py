import numpy as np

from sgf_utils.sgf_parser import SGFNode, Move

number_to_color = {1: "B", -1: "W"}


def add_stones_to_node(node: SGFNode, array):
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            value = int(array[i, j])
            if value != 0:
                try:
                    color = number_to_color[value]
                except Exception as e:
                    print(e)
                node.add_list_property(f"A{color}", [Move(coords=(i, j), player=color).sgf(node.board_size)])


def detect_move(board_state1, board_state2):
    moves = np.where(np.logical_and(board_state1 == 0, board_state2 != 0))
    coords = list(zip(*moves))
    if len(coords) == 0:
        return None
    elif len(coords) == 1:
        return coords[0], board_state2[coords[0]]
    else:
        raise ValueError("More than one move detected")
