import os
import random

from sgf_utils.game import BaseGame, KaTrainSGF
from sgf_utils.game_node import GameNode
from sgf_utils.engine import KataGoEngine
from sgf_utils.base_katrain import KaTrainBase
import json
from simplify_log import simplify_log

config_path = r"../config.json"

with open(config_path, 'r') as f:
    config = json.load(f)
katrain = KaTrainBase()


def get_stones_on_board(sgf_string, engine, game_name, percentage_to_go_back=0.15, randomize_percentage=True, number_of_position_to_check=1):
    def analyse_node(node, move_num, game_name, players='BW'):
        try:
            game = BaseGame(node)
        except Exception as e:
            print(e)
            print(sgf_string)
            return
        stones = game.stones
        white_sgf_stones = [s.sgf(game.board_size) for s in stones if s.player == 'W']
        black_sgf_stones = [s.sgf(game.board_size) for s in stones if s.player == 'B']
        for player in players:
            position_node = GameNode(properties={'AW': white_sgf_stones, 'AB': black_sgf_stones, 'SZ': ':'.join(map(str, game.board_size))}, player=player)
            position_node.analyze(engine, visits=1, query_id=f'G{game_name}M{move_num}{player}')

    root = KaTrainSGF(sgf_string).root
    node = root
    game_len = 0

    analysed_positions = False
    while True:
        node = node.children[0]
        game_len += 1
        move_num = game_len
        if not analysed_positions and (node.is_pass or not node.children):
            node_to_analyse = node
            # print(f"Game with {game_len} moves, pass = {node.is_pass}")
            percentage_to_go_back_this_time = percentage_to_go_back
            if randomize_percentage:
                percentage_to_go_back_this_time *= random.random()
            go_back_max_this_many_moves = int(percentage_to_go_back_this_time * game_len) + 1
            check_every_nth_position = go_back_max_this_many_moves // number_of_position_to_check
            if check_every_nth_position == 0:
                # The game is very short, we don't want to take a look at it
                break
            for i in range(number_of_position_to_check):
                for _ in range(check_every_nth_position):
                    move_num -= 1
                    node_to_analyse = node_to_analyse.parent
                # print(f'Went back {check_every_nth_position}')
                analyse_node(node_to_analyse, move_num=move_num, game_name=game_name, players='BW')
                # print(f"Analyzing position {i}")
            analysed_positions = True
        if not node.children:
            analyse_node(node, move_num="-last-", game_name=game_name, players='B')
            break


def analyze_sgfs(filename, logfile=None):
    eng = KataGoEngine(katrain, config["engine"], custom_logfile=logfile)
    # eng.write_stdin_thread.join()
    with open(filename, 'r') as f:
        for counter, sgf_string in enumerate(f, 1):
            get_stones_on_board(sgf_string, eng, game_name=str(counter))
    eng.shutdown(finish=True)
    # eng.write_queue.put(None)
    # for t in [eng.write_stdin_thread, eng.analysis_thread, eng.stderr_thread]:
    #     if t:
    # eng.analysis_thread.join()
    # exit_code = eng.katago_process.wait()
    # while not eng.write_queue.empty():
    #     time.sleep(2)
    print(f'Finished, sent {counter} requests')


PROJECT_ROOT = os.getcwd()
selfplay_dir = os.path.join(PROJECT_ROOT, "kata-selfplay")
refined_log_dir = os.path.join(PROJECT_ROOT, "refined_logs")
log_dir = os.path.join(PROJECT_ROOT, "analysis_logs")


def process_selfplay(folder):
    batchpath = os.path.join(selfplay_dir, folder, 'sgfs')
    # Assume that batchpath is a directory
    log_folder = os.path.join(log_dir, folder)
    refined_log_folder = os.path.join(refined_log_dir, folder)
    for sgfs_file in os.listdir(batchpath):
        if sgfs_file.endswith('.sgfs'):
            log_filename = sgfs_file[:-5] + ".log"
            refined_logfile = os.path.join(refined_log_folder, log_filename)
            if not os.path.exists(refined_logfile):
                sgfs_path = os.path.join(batchpath, sgfs_file)
                os.makedirs(log_folder, exist_ok=True)
                os.makedirs(refined_log_folder, exist_ok=True)
                logfile = os.path.join(log_folder, log_filename)
                analyze_sgfs(sgfs_path, logfile=logfile)
                print(f'Analyzed {folder}: {sgfs_file}!')
                simplify_log(logfile, refined_logfile, sgfs_path)
                assert os.path.exists(refined_logfile), f"Refined logfile {folder}/{log_filename} not created yet!"
                os.unlink(logfile)

# input()
# sgf_string_illegal = """(;FF[4]GM[1]SZ[19]PB[b20c256-s344142336-d171075566]PW[b20c256-s344142336-d171075566]HA[3]KM[17]RU[koSIMPLEscoreAREAtaxNONEsui1button1]RE[B+3.5]AB[cd][gf][pp]C[startTurnIdx=277,mode=0,modeM1=0,modeM2=1,newNeuralNetTurn345=b20c256-s344987392-d171300566];B[qd];W[cp];B[fc];W[ee];B[fd];W[oc];B[mc];W[pf];B[od];W[nd];B[oe];W[nc];B[pc];W[dd];B[ce];W[dc];B[jc])"""


