import argparse
import json
import os
import time
import traceback
from pathlib import Path

import yaml
from tqdm import tqdm

from common.utils import get_ground_truth, almost_equal
from game_tree.local_position_node import LocalPositionNode, LocalPositionSGF
from game_tree.position_tree import PositionTree

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = '.79'



def find_temp(gtp_position, config, max_depth=20, output_sgf_path=None, output_json_path=None,
              initialized_engines=None, verbose=True):
    game_name = os.path.basename(gtp_position)
    root_node: LocalPositionNode = LocalPositionSGF.parse_file(gtp_position)
    initialized_engines = initialized_engines or {}
    print(f'Processing {game_name}')
    position_tree = PositionTree(root_node, config=config, game_name=game_name, **initialized_engines)
    print(f'Processing {game_name}')
    gt = get_ground_truth(position_tree.root.get_property("C"))
    print(f'Processing {game_name}')
    value = None
    temp = None
    not_ok_game = ""
    try:
        position_tree.build_tree(max_depth=max_depth, reset_engine=True, delete_engine=False, verbose=verbose)
        print("Built tree")
        temp = float(position_tree.root.cgt_game.temp)
        value = 2 * temp - 2
    except Exception as e:
        print(e)
        if 'is not OK' in str(e):
            if verbose:
                print(str(e))
            not_ok_game = str(e)
        else:
            if verbose:
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
            if verbose:
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


def get_positions(input_path):
    positions = []
    if input_path.endswith(".sgf"):
        positions = [input_path]
    elif os.path.isdir(input_path):
        positions = []
        for folder, subfolders, files in os.walk(input_path):
            for file in files:
                if file.endswith(".sgf"):
                    positions.append(os.path.join(folder, file))
    return positions


def run_benchmark(positions, config, max_depth, position_root_dir, output_basename, output_root_dir='assets', verbose=True):
    initialized_engines: dict = None

    try:
        start_time = time.time()
        output_dir = position_root_dir if os.path.isdir(position_root_dir) else os.path.dirname(position_root_dir)
        output_dir = os.path.join(output_root_dir, 'results', os.path.basename(output_dir), output_basename)
        output_filepath = os.path.join(output_dir, 'results.yaml')
        results_per_pos = {}
        error_count = 0
        correct_count = 0
        no_gt_count = 0
        tqdm_bar = tqdm(positions) if verbose else positions
        for position in tqdm_bar:
            try:
                output_sgf = os.path.join(output_dir, 'analyzed',
                                          os.path.basename(os.path.dirname(position)) + '-' + os.path.basename(
                                              position))
                output_json = os.path.join(output_dir, 'analyzed_json',
                                           os.path.basename(os.path.dirname(position)) + '-' +
                                           os.path.splitext(os.path.basename(position))[0] + '.json')
                print(f'Processing {position}')
                gt, temp, iter_num, inferenece_num, depth, initialized_engines = find_temp(position, config,
                                                                                           max_depth,
                                                                                           output_sgf, output_json,
                                                                                           initialized_engines,
                                                                                           verbose=verbose)
                print(f'GT: {gt}, Temp: {temp}, Iter: {iter_num}, Inference: {inferenece_num}, Depth: {depth}')
                filename = str(Path(position).relative_to(position_root_dir))
                cur_results = {'temp': temp, 'iterations': iter_num,
                               'depth': depth}  # , 'inferences': inferenece_num}
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
        if verbose:
            print(f'Processing of {len(positions)} positions took {round(time.time() - start_time)}s')
        return all_results
    except Exception:
        traceback.print_exc()
        return None
    finally:
        if initialized_engines is not None:
            keys = list(initialized_engines.keys())
            for key in keys:
                engine = initialized_engines.pop(key)
                engine.shutdown()


def find_configs(config_arg):
    configs = config_arg.split(',')
    configs = [os.path.join('analysis_config', conf_file) for conf_file in configs]
    assert all(os.path.exists(conf_file) for conf_file in configs), 'Specify existing config filenames!'
    return configs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Path to the input directory/file')  # bachelor-value-masked
    # r"models/conv1x1-pretr-0405b-102-before-unfreezing.ckpt", , "models/conv1x1-pretr-unfrozen_from_start.ckpt"
    parser.add_argument('--config', type=str, default=r"a0_thresholded.yaml",
                        help='Config name or comma-separated names')
    parser.add_argument('--a0ckpt', type=str, default=None,
                        help='If not None, uses the specified checkpoint instead of the one from the config file')
    parser.add_argument('--max_depth', type=int, default=15, help='Max tree depth')
    args = parser.parse_args()
    positions = get_positions(args.input)
    configs = find_configs(args.config)
    for conf_file in configs:
        with open(conf_file, 'r') as f:
            config = yaml.safe_load(f)
        output_basename = os.path.basename(conf_file).rsplit('.', 1)[0] + '-results'
        if args.a0ckpt is not None:
            config['evaluator_kwargs']['a0_ckpt'] = args.a0ckpt
            output_basename += f'-a0ckpt-{os.path.basename(args.a0ckpt).rsplit(".", 1)[0]}'
        run_benchmark(positions, config, args.max_depth, args.input, output_basename)
