import os

from game import BaseGame, KaTrainSGF
from game_node import GameNode

# import argparse
#
# parser = argparse.ArgumentParser()
# parser.add_argument('--directory', '-d', type=str)
# directory = parser.parse_args().directory


def get_stones_on_board(sgf_string, folder, file, seek_comments=True, add_players_info_in_comments=False):
    def get_stones_from_node(node):
        try:
            game = BaseGame(node)
        except Exception as e:
            print(sgf_string)
            raise e
        stones = game.stones
        white_sgf_stones = [s.sgf(game.board_size) for s in stones if s.player == 'W']
        black_sgf_stones = [s.sgf(game.board_size) for s in stones if s.player == 'B']
        position_node = GameNode(properties={'AW': white_sgf_stones, 'AB': black_sgf_stones}) #, 'SZ': ':'.join(map(str, game.board_size))})
        position_node.children = node.children
        return position_node

    root = KaTrainSGF(sgf_string).root
    if seek_comments:
        node = root
        while node.children:
            node = node.children[0]
            if node.get_property('C'):
                single_position = get_stones_from_node(node)
                break
        else:
            single_position = get_stones_from_node(root)

    else:
        single_position = get_stones_from_node(root)

    if add_players_info_in_comments:
        single_position.properties['C'] = f'Black: {root.get_property("PB")} vs White: {root.get_property("PW")}\n{single_position.get_property("C")}'

    return single_position

# sgf_string_illegal = """(;FF[4]GM[1]SZ[19]PB[b20c256-s344142336-d171075566]PW[b20c256-s344142336-d171075566]HA[3]KM[17]RU[koSIMPLEscoreAREAtaxNONEsui1button1]RE[B+3.5]AB[cd][gf][pp]C[startTurnIdx=277,mode=0,modeM1=0,modeM2=1,newNeuralNetTurn345=b20c256-s344987392-d171300566];B[qd];W[cp];B[fc];W[ee];B[fd];W[oc];B[mc];W[pf];B[od];W[nd];B[oe];W[nc];B[pc];W[dd];B[ce];W[dc];B[jc])"""

# Example usage
# stones = get_stones_on_board(sgf_string)
# print(stones)


root_path = input("What directory to merge?\n")
if root_path.startswith('"'):
    root_path = root_path[1:]
if root_path.endswith('"'):
    root_path = root_path[:-1]
root_node = GameNode(properties={'SZ': 19})
root_node.children.append(GameNode(properties={'N': ''}))
seek_comments = input("Encode moves until the first comment as a position? (y/n)\n").lower() == 'y'
add_players_info_in_comments = input("Add players info in comments? (y/n)\n").lower() == 'y'
try:
    for file in os.listdir(root_path):
        if not file.endswith('.sgf'):
            continue
        filepath = os.path.join(root_path, file)
        with open(filepath, 'r', errors='ignore') as f:
            sgf_string = f.read()
        root_node.children.append(get_stones_on_board(sgf_string, root_path, file, seek_comments=seek_comments, add_players_info_in_comments=add_players_info_in_comments))
        print(f'Added {file}!')

    merged_game = BaseGame(root_node)
    merged_game.write_sgf(os.path.join(root_path, 'merged.sgf'))
    input("Merged game written to merged.sgf. Press enter to exit.")
except Exception as e:
    print(e)
    input("Program has failed. Press enter to exit.")
