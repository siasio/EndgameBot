import os
from typing import Optional, Tuple, List

from game import BaseGame, KaTrainSGF
from game_node import GameNode
from sgf_parser import SGFNode


# import argparse
#
# parser = argparse.ArgumentParser()
# parser.add_argument('--directory', '-d', type=str)
# directory = parser.parse_args().directory


def get_stones_on_board(sgf_string, folder, file, seek_comments=True, seek_second_comment=True, add_players_info_in_comments=False):
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
        seeking_first_comment = True
        node = root
        coords_list: List[Optional[Tuple[int, int]]] = []
        single_position = None
        probable_position = None
        checking_root = True
        while node.children:
            if checking_root:
                checking_root = False
            else:
                node: SGFNode = node.children[0]
            if seek_second_comment and not seeking_first_comment:
                coords_list.append(node.move.sgf(node.board_size) if node.move else None)
            if node.get_property('C'):
                if not seek_second_comment:
                    single_position = get_stones_from_node(node)
                    break
                elif seeking_first_comment:
                    probable_position = get_stones_from_node(node)
                    seeking_first_comment = False
                else:
                    single_position = get_stones_from_node(node)
                    for i, coords in enumerate(coords_list, 1):
                        if coords is not None:
                            single_position.add_list_property('LB', [f'{coords}:{i}'])
                    break
        else:
            if probable_position:
                # Only one comment was found despite seeking two
                single_position = probable_position
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
seek_second_comment = input("Encode moves until the second comment as numbered stones? (y/n)\n").lower() == 'y'
add_players_info_in_comments = input("Add players info in comments? (y/n)\n").lower() == 'y'
merge_games = input("Merge games? (y/n)\n").lower() == 'y'
try:
    for file in os.listdir(root_path):
        if not file.endswith('.sgf'):
            continue
        filepath = os.path.join(root_path, file)
        with open(filepath, 'r', errors='ignore') as f:
            sgf_string = f.read()
        new_node = get_stones_on_board(sgf_string, root_path, file, seek_comments=seek_comments, seek_second_comment=seek_second_comment, add_players_info_in_comments=add_players_info_in_comments)
        if not merge_games:
            new_game = BaseGame(new_node)
            new_game.write_sgf(os.path.join(root_path, file.rsplit('.', 1)[0] + '_pos.sgf'))
            continue
        root_node.children.append(new_node)
        print(f'Added {file}!')

    if merge_games:
        merged_game = BaseGame(root_node)
        merged_game.write_sgf(os.path.join(root_path, 'merged.sgf'))
        input("Merged game written to merged.sgf. Press enter to exit.")
    else:
        input("Games written. Press enter to exit.")
except Exception as e:
    print(e)
    input("Program has failed. Press enter to exit.")
