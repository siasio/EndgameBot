# Evaluate positions just as in visualize.py but without a GUI

import argparse
import datetime
import json
import os
import pickle
import traceback

import cv2
import numpy as np
import yaml
from cloudpickle import cloudpickle

from common.utils import lowest_number_for_dir
from legacy.build_tree import PositionTree
from game_tree.local_position_node import LocalPositionSGF, LocalPositionNode
# from katago_wrapper import KatagoWrapper
from legacy.local_pos_masks import AnalyzedPosition
from policies.resnet_policy import ResnetPolicyValueNet128, TransferResnet
from sgf_utils.sgf_parser import SGFNode, Move
from game_tree.position_tree import PositionTree as PT


# from cgt_engine.cgt_engine import find_mt3


def load_agent(ckpt_path):
    if ckpt_path == "katago":
        return KatagoWrapper()
    backbone = ResnetPolicyValueNet128(input_dims=(9, 9, 9), num_actions=82)
    agent = TransferResnet(backbone)
    agent = agent.eval()
    with open(ckpt_path, "rb") as f:
        loaded_agent = pickle.load(f)
        if "agent" in loaded_agent:
            loaded_agent = loaded_agent["agent"]
        agent = agent.load_state_dict(loaded_agent)
    return agent


def detect_mask(a0pos, agent):
    a0pos.local_mask = [[1 for y in range(a0pos.pad_size)]
                                for x in range(a0pos.pad_size)]
    a0pos.board_mask = [[1 for y in range(a0pos.pad_size)]
                                      for x in range(a0pos.pad_size)]
    a0pos.analyze_and_decompose(a0pos.local_mask, agent)
    # print(a0pos.segmentation)
    biggest_segment = None
    biggest_segment_size = 0
    for i in range(1, np.max(a0pos.segmentation) + 1):
        segment = a0pos.segmentation == i
        segment_size = np.sum(segment)
        if biggest_segment_size < segment_size < 20:
            biggest_segment_size = segment_size
            biggest_segment = segment
    if biggest_segment is not None:
        biggest_segment = cv2.dilate(biggest_segment.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
    a0pos.local_mask = biggest_segment


def add_mask_to_node(node: SGFNode, array):
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            value = array[i, j]
            if value == 0:
                node.add_list_property("TR", [Move(coords=(i, j)).sgf(node.board_size)])


def time_now():
    return datetime.datetime.now().strftime('%H:%M:%S')


def visualize_position(gtp_position, output_dir, agent=None, from_pkl=False, from_sgf=False, from_json=False, mask_from_sgf=False, cgt_engine=False):
    gt_temperature, gt_score_b, gt_score_w = None, None, None
    game_name = None
    if from_sgf:
        game_name = os.path.basename(gtp_position)
        with open(gtp_position, 'r') as f:
            sgf = f.read()
        a0pos: AnalyzedPosition = AnalyzedPosition.from_sgf_file(sgf, mask_from_sgf=mask_from_sgf)
        gt_data = a0pos.game.root.get_property("C")
        if gt_data:
            gt_data = gt_data.splitlines()
            for line in gt_data:
                if "T" in line:
                    gt_temperature = float(line[2:])
                if "S" in line:
                    gt_score_b, gt_score_w = line[2:].split(" ")
                    gt_score_b = float(gt_score_b)
                    gt_score_w = float(gt_score_w)
        a0pos.board_mask = np.ones_like(a0pos.board_mask)
        # a0pos.board_mask[:4, 16:] = 1
    elif from_json:
        a0pos: AnalyzedPosition = AnalyzedPosition.from_gtp_log(json.loads(gtp_position))
    elif from_pkl:
        a0pos: AnalyzedPosition = AnalyzedPosition.from_jax(gtp_position)
    else:
        return

    if not mask_from_sgf:
        detect_mask(a0pos, agent)
        if a0pos.local_mask is None:
            print(f"No local positions found in {gtp_position}")
            return

    position_tree: PositionTree = PositionTree.from_a0pos(a0pos, game_name=game_name)
    position_tree.load_agent(agent)
    position_tree.max_depth = 20

    position_tree.real_run(cgt_engine=cgt_engine, multiple_threshold=0.1)
    if cgt_engine:
        for node in position_tree.iter_nodes():
            try:
                cgt_info = node.cgt_info()
            except Exception as e:
                cgt_info = "CGT calculations failed"
            node.set_property("C", cgt_info)
        print(position_tree.root.get_property("C"))

    # stones = position_tree.get_position()
    # print(position_tree.position_as_string(stones))
    # print(position_tree.position_as_string(a0pos.local_mask))
    # print(f"[{gtp_position}] Temperature {position_tree.current_node.temperature}")
    # node = SGFNode()
    else:
        temperatures = [f'Depth {i}: T {t:.2f} CN {cn} CS {cs}' for i, (t, cn, cs) in enumerate(zip(position_tree.temperatures, position_tree.checked_nodes, position_tree.score_stats))]
        temperature_str = '\n'.join(temperatures)
        position_tree.root.set_property("C", f"Temperature / Checked nodes / Children scores:\n{temperature_str}")
    add_mask_to_node(position_tree.root, a0pos.local_mask)
    # add_stones_to_node(node, stones)
    # add_mask_to_node(node, a0pos.local_mask)
    filename = os.path.basename(gtp_position) if os.path.isfile(gtp_position) else str(lowest_number_for_dir(output_dir)).zfill(3) + '.sgf'
    filepath = os.path.join(output_dir, filename)

    position_tree.write_sgf(filepath)

    return
    try:

        t_1 = position_tree.temperatures[0]
        score_stats_1 = position_tree.score_stats[0]
        t_final = position_tree.temperatures[-1]
        score_stats_final = position_tree.score_stats[-1]
        if gt_temperature is not None and gt_score_b is not None and gt_score_w is not None:
            print(f"{os.path.basename(output_dir)} - {filename}")
            print(f"GT temperature: {gt_temperature}, initial estimation: {t_1}, final temperature: {t_final}")
            print(f"GT score: {gt_score_b} - {gt_score_w}, initial estimation: {score_stats_1}, final estimation: {score_stats_final}")
    except IndexError:
        print(f"{os.path.basename(output_dir)} - {filename}")
        print("Didn't expand the tree!")
    # with open(, 'w') as f:
    #     f.write(node.sgf())


def new_vis(gtp_position, config, output_dir):
    game_name = os.path.basename(gtp_position)
    root_node: LocalPositionNode = LocalPositionSGF.parse_file(gtp_position)
    position_tree = PT(root_node, config=config, game_name=game_name)
    try:
        position_tree.build_tree(max_depth=20)
    except Exception as e:
        traceback.print_exc()
    for node in position_tree.iter_nodes():
        try:
            cgt_info = node.cgt_info()
        except Exception as e:
            traceback.print_exc()
            cgt_info = "CGT calculations failed"
        node.set_property("C", cgt_info)
    print(position_tree.root.get_property("C"))
    add_mask_to_node(position_tree.root, position_tree.mask)
    filename = os.path.basename(gtp_position) if os.path.isfile(gtp_position) else str(
        lowest_number_for_dir(output_dir)).zfill(3) + '.sgf'
    filepath = os.path.join(output_dir, filename)

    position_tree.write_sgf(filepath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Path to the input file') # bachelor-value-masked
    # parser.add_argument('--output', type=str, default=r"C:\Users\StanislawFrejlak\Uni\masters\strange\frozen-cutoff", help='Path to the output directory')
    # r"models/conv1x1-pretr-0405b-102-before-unfreezing.ckpt", , "models/conv1x1-pretr-unfrozen_from_start.ckpt"
    parser.add_argument('--model', type=str, nargs='+', default=[r"models\conv1x1-pretr-0405b-final.ckpt"], help='Path to the model file')
    parser.add_argument('--config', type=str, default=r"a0_thresholded.yaml", help='Config name')

    args = parser.parse_args()
    from_pkl = False
    from_json = False
    from_sgf = False
    positions = []
    selected_sgf = args.input
    if selected_sgf.endswith(".log"):
        from_json = True
        with open(selected_sgf, 'r') as f:
            positions = f.read().splitlines()
    elif selected_sgf.endswith(".pkl"):
        from_pkl = True
        with open(selected_sgf, "rb") as f:
            positions = cloudpickle.load(f)
            positions = positions[:100]

    elif selected_sgf.endswith(".sgf"):
        from_sgf = True
        positions = [selected_sgf]
    elif os.path.isdir(selected_sgf):
        from_sgf = True
        positions = []
        for file in os.listdir(selected_sgf):
            if file.endswith(".sgf"):
                positions.append(os.path.join(selected_sgf, file))

    if args.config is not None:
        config_file = os.path.join('../analysis_config', args.config)
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        output_dir = selected_sgf if os.path.isdir(selected_sgf) else os.path.dirname(selected_sgf)
        output_dir = os.path.join(output_dir, args.config.rsplit('.', 1)[0] + 'new')
        os.makedirs(output_dir, exist_ok=True)
        for position in positions:
            try:
                new_vis(position, config, output_dir)
            except Exception:
                traceback.print_exc()
    else:
        models_names = args.model
        models_to_check = [load_agent(model) for model in models_names]

        for agent, model_name in zip(models_to_check, models_names):
            output_dir = selected_sgf if os.path.isdir(selected_sgf) else os.path.dirname(selected_sgf)
            output_dir = os.path.join(output_dir, os.path.splitext(os.path.basename(model_name))[0])
            os.makedirs(output_dir, exist_ok=True)
            for position in positions:
                try:
                    visualize_position(position, output_dir=output_dir, agent=agent, from_pkl=from_pkl, from_sgf=from_sgf, from_json=from_json, mask_from_sgf=True, cgt_engine=True)
                except Exception as e:
                    print(f"Error while processing {position}:")
                    print(traceback.format_exc())
                    # raise e
    