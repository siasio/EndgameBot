# Evaluate positions just as in visualize.py but without a GUI

import argparse
import json
import os
import pickle
import sys
import time
from typing import List, Tuple

import cv2
import numpy as np
from cloudpickle import cloudpickle

from build_tree import PositionTree
from katago_wrapper import KatagoWrapper
from local_pos_masks import AnalyzedPosition
from policies.resnet_policy import ResnetPolicyValueNet128, TransferResnet
from sgf_parser import SGFNode, Move
from array_katrain_utils import add_stones_to_node


def load_agent(ckpt_path):
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
        if biggest_segment_size < segment_size and segment_size < 20:
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


def lowest_number_for_dir(dir):
    if not os.path.exists(dir):
        return 0
    basenames = [f.split(".")[0] for f in os.listdir(dir)]
    numbers = [int(b) for b in basenames if b.isnumeric()]
    return max(numbers) + 1 if numbers else 0


def visualize_position(gtp_position, output_dir, agent=None, from_pkl=False, from_sgf=False, from_json=False, mask_from_sgf=False):
    if from_sgf:
        with open(gtp_position, 'r') as f:
            sgf = f.read()
        a0pos: AnalyzedPosition = AnalyzedPosition.from_sgf_file(sgf, mask_from_sgf=mask_from_sgf)
        a0pos.board_mask = np.ones_like(a0pos.board_mask)
        # a0pos.board_mask[:4, 16:] = 1
    elif from_json:
        a0pos: AnalyzedPosition = AnalyzedPosition.from_gtp_log(json.loads(gtp_position))
    elif from_pkl:
        a0pos: AnalyzedPosition = AnalyzedPosition.from_jax(gtp_position)
    else:
        return
    position_tree: PositionTree = PositionTree.from_a0pos(a0pos)
    
    # a0pos.local_mask = a0pos.local_mask if a0pos.fixed_mask else a0pos.board_mask

    if not mask_from_sgf:
        detect_mask(a0pos, agent)
        if a0pos.local_mask is None:
            print(f"No local positions found in {gtp_position}")
            return
    position_tree.load_agent(agent)
    position_tree.max_depth = 5
    # a0pos.analyze_pos(local_mask, agent)
    position_tree.run()
    stones = position_tree.get_position()
    print(position_tree.position_as_string(stones))
    print(position_tree.position_as_string(a0pos.local_mask))
    print(f"[{gtp_position}] Temperature {position_tree.current_node.temperature}")
    # node = SGFNode()
    temperatures = [f'Depth {i}: T {t:.2f} CN {cn} CS {cs}' for i, (t, cn, cs) in enumerate(zip(position_tree.temperatures, position_tree.checked_nodes, position_tree.score_stats))]
    temperature_str = '\n'.join(temperatures)
    position_tree.root.set_property("C", f"Temperature / Checked nodes / Children scores:\n{temperature_str}")
    add_mask_to_node(position_tree.root, a0pos.local_mask)
    # add_stones_to_node(node, stones)
    # add_mask_to_node(node, a0pos.local_mask)
    filename = os.path.basename(gtp_position) if os.path.isfile(gtp_position) else str(lowest_number_for_dir(output_dir)).zfill(3) + '.sgf'
    filepath = os.path.join(output_dir, filename)

    position_tree.write_sgf(filepath)
    # with open(, 'w') as f:
    #     f.write(node.sgf())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=r"C:\Users\StanislawFrejlak\Uni\masters\bachelor-value-masked", help='Path to the input file')
    parser.add_argument('--output', type=str, default=r"C:\Users\StanislawFrejlak\Uni\masters\bachelor-value-masked\frozen", help='Path to the output directory')
    parser.add_argument('--model', type=str, default=r"C:\Users\StanislawFrejlak\Documents\GitHub\EndgameBot\a0-jax\trained_2023-12-frozen.ckpt", help='Path to the model file')
    
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
        position = [selected_sgf]
        # with open(selected_sgf, 'r') as f:
        #     positions = [f.read()]
    elif os.path.isdir(selected_sgf):
        from_sgf = True
        positions = []
        for file in os.listdir(selected_sgf):
            if file.endswith(".sgf"):
                positions.append(os.path.join(selected_sgf, file))

    os.makedirs(args.output, exist_ok=True)
    agent = load_agent(args.model) if args.model else KatagoWrapper()
    # agent = KatagoWrapper()
    for position in positions:
        try:
            visualize_position(position, output_dir=args.output, agent=agent, from_pkl=from_pkl, from_sgf=from_sgf, from_json=from_json, mask_from_sgf=True)
        except Exception as e:
            print(f"Error while processing {position}: {e}")
            # raise e
    