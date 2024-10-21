import argparse
import json
import os
import time
import traceback

import yaml
from tqdm import tqdm

from common.utils import get_ground_truth, almost_equal
from game_tree.local_position_node import LocalPositionNode, LocalPositionSGF
from game_tree.position_tree import PositionTree


def find_temp(gtp_position, config, max_depth=20, output_sgf_path=None, output_json_path=None, initialized_engines=None):
    game_name = os.path.basename(gtp_position)
    root_node: LocalPositionNode = LocalPositionSGF.parse_file(gtp_position)
    initialized_engines = initialized_engines or {}
    position_tree = PositionTree(root_node, config=config, game_name=game_name, **initialized_engines)
    gt = get_ground_truth(position_tree.root.get_property("C"))
    value = None
    temp = None
    not_ok_game = ""
    try:
        position_tree.build_tree(max_depth=max_depth, reset_engine=True, delete_engine=False)
        temp = float(position_tree.root.cgt_game.temp)
        value = 2 * temp - 2
    except Exception as e:
        if 'is not OK' in str(e):
            print(str(e))
            not_ok_game = str(e)
        else:
            traceback.print_exc()
    finally:
        if output_sgf_path is not None:
            for node in position_tree.iter_nodes():
                try:
                    cgt_info = node.cgt_info()
                except Exception as e:
                    traceback.print_exc()
                    cgt_info = "CGT calculations failed " + not_ok_game
                node.set_property("C", cgt_info)
            print(position_tree.root.get_property("C"))
            # add_mask_to_node(position_tree.root, position_tree.mask)

            position_tree.write_sgf(output_sgf_path)
        if output_json_path is not None:
            os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
            nodes_json_info = [node.as_json() for node in position_tree.iter_nodes()]
            json_info = {
                'nodes': nodes_json_info,
                'ground_truth_temp': gt / 2 + 1,
                'calculated_temp': temp,
            }
            with open(output_json_path, 'w') as f:
                json.dump(json_info, f)
        iter_num = position_tree.iter_num
        inference_num = position_tree.inference_num
        depth = position_tree.root.expanded_tree_depth
        initialized_engines = position_tree.eval.evaluator_registry
        return gt, value, iter_num, inference_num, depth, initialized_engines


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Path to the input directory/file') # bachelor-value-masked
    # r"models/conv1x1-pretr-0405b-102-before-unfreezing.ckpt", , "models/conv1x1-pretr-unfrozen_from_start.ckpt"
    parser.add_argument('--config', type=str, default=r"a0_thresholded.yaml", help='Config name or comma-separated names')
    parser.add_argument('--max_depth', type=int, default=10, help='Max tree depth')

    args = parser.parse_args()

    start_time = time.time()

    positions = []
    selected_sgf = args.input

    if selected_sgf.endswith(".sgf"):
        positions = [selected_sgf]
    elif os.path.isdir(selected_sgf):
        positions = []
        for folder, subfolders, files in os.walk(selected_sgf):
            for file in files:
                if file.endswith(".sgf"):
                    positions.append(os.path.join(folder, file))
    configs = args.config.split(',')
    configs = [os.path.join('analysis_config', conf_file) for conf_file in configs]
    assert all(os.path.exists(conf_file) for conf_file in configs), 'Specify existing config filenames!'
    initialized_engines: dict = None
    for conf_file in configs:
        if initialized_engines is not None:
            keys = list(initialized_engines.keys())
            for key in keys:
                engine = initialized_engines.pop(key)
                engine.shutdown()

        initialized_engines = None
        try:
            with open(conf_file, 'r') as f:
                config = yaml.safe_load(f)
            output_dir = selected_sgf if os.path.isdir(selected_sgf) else os.path.dirname(selected_sgf)
            output_dir = os.path.join(os.path.dirname(output_dir), 'results', os.path.basename(output_dir), os.path.basename(conf_file).rsplit('.', 1)[0])
            output_filepath = os.path.join(output_dir, os.path.basename(conf_file).rsplit('.', 1)[0] + '-results.yaml')
            results_per_pos = {}
            error_count = 0
            correct_count = 0
            no_gt_count = 0
            for position in tqdm(positions):
                try:
                    output_sgf = os.path.join(output_dir, 'analyzed', os.path.basename(os.path.dirname(position)) + '-' + os.path.basename(position))
                    output_json = os.path.join(output_dir, 'analyzed_json', os.path.basename(os.path.dirname(position)) + '-' + os.path.splitext(os.path.basename(position))[0] + '.json')
                    gt, temp, iter_num, inferenece_num, depth, initialized_engines = find_temp(position, config, args.max_depth, output_sgf, output_json, initialized_engines)
                    filename = position[len(selected_sgf) + 1:]
                    cur_results = {'temp': temp, 'iterations': iter_num, 'depth': depth} #, 'inferences': inferenece_num}
                    if gt is not None:
                        if temp is None:
                            error_count += 1
                        cur_results['gt'] = gt
                        cur_results['correct'] = temp is not None and almost_equal(temp, gt)
                        if cur_results['correct']:
                            correct_count += 1
                    else:
                        no_gt_count += 1
                    results_per_pos[filename] = cur_results
                except Exception:
                    traceback.print_exc()
            all_results = {
                'num_positions': len(positions),
                'correct_count': correct_count,
                'error_count': error_count,
                'no_gt_count': no_gt_count,
                'results_per_pos': results_per_pos
            }
            with open(output_filepath, 'w') as f:
                yaml.dump(all_results, f)
            print(f'Processing of {len(positions)} positions with {conf_file} took {round(time.time() - start_time)}s')
        except Exception:
            traceback.print_exc()
