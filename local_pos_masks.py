from sgf_parser import Move
import numpy as np
import jax.numpy as jnp
import jax
from game import BaseGame, KaTrainSGF
from utils import stack_last_state
import time
import cv2
import random
from scipy.ndimage.measurements import label
#from train_agent import TrainingOwnershipExample
import os

#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = '.79'

BLACK = 1
WHITE = -1
ownership_strict_threshold = 5
ownership_lenient_threshold = 15
max_ownership = 100

kernel_cv = np.zeros((5, 5), dtype=np.int8)
kernel_cv[:, 2] = 1
kernel_cv[2, :] = 1
kernel_cv[1, 1] = 1
kernel_cv[1, 3] = 1
kernel_cv[3, 1] = 1
kernel_cv[3, 3] = 1

kernel_small = np.zeros((3, 3), dtype=np.uint8)
kernel_small[:, 1] = 1
kernel_small[1, :] = 1

kernel_medium = np.ones((3, 3), dtype=np.uint8)

artificial_kernels = [
    np.array([[0, 1, 1, 1, 1], [1, 1, 1, 0, 0], [1, 1, 1, 1, 0], [0, 1, 1, 0, 1], [0, 0, 1, 0, 0]], dtype=np.uint8),
    np.array([[1, 0, 0, 0, 0], [1, 1, 1, 1, 1], [1, 0, 1, 0, 1], [1, 1, 0, 1, 0], [0, 1, 0, 0, 0]], dtype=np.uint8),
    np.array([[0, 0, 0, 1, 1], [0, 0, 1, 1, 1], [0, 1, 1, 1, 0], [0, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=np.uint8),
    np.array([[0, 0, 0, 0, 1], [1, 1, 1, 0, 1], [0, 0, 1, 1, 1], [0, 1, 1, 0, 0], [1, 0, 1, 1, 0]], dtype=np.uint8),
    np.array([[0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 1], [0, 1, 1, 1, 1], [1, 1, 1, 0, 1]], dtype=np.uint8),
    np.array([[0, 0, 0, 1, 0], [1, 1, 1, 1, 0], [0, 0, 1, 1, 0], [0, 1, 1, 1, 0], [1, 1, 1, 1, 0]], dtype=np.uint8),
    np.array([[0, 1, 0, 0, 0], [0, 1, 1, 0, 0], [0, 1, 1, 1, 1], [0, 1, 1, 1, 1], [1, 1, 0, 0, 0]], dtype=np.uint8),
    np.array([[0, 0, 1, 1, 0], [1, 0, 1, 1, 0], [0, 0, 1, 1, 1], [0, 0, 1, 1, 0], [1, 0, 0, 0, 0]], dtype=np.uint8),
    np.array([[0, 1, 1, 0, 0], [0, 0, 1, 0, 0], [0, 1, 1, 0, 0], [0, 1, 1, 0, 1], [0, 0, 1, 1, 1]], dtype=np.uint8),
    np.array([[0, 1, 1, 1, 1], [1, 1, 1, 0, 0], [1, 0, 1, 1, 0], [1, 1, 1, 1, 0], [0, 0, 0, 1, 1]], dtype=np.uint8),
    np.array([[0, 1, 1, 0, 1], [0, 1, 1, 0, 0], [0, 1, 1, 0, 0], [1, 1, 1, 1, 1], [0, 1, 1, 1, 1]], dtype=np.uint8),
    np.array([[1, 1, 1, 1, 1], [1, 0, 1, 1, 0], [1, 0, 1, 0, 0], [1, 1, 0, 0, 0], [0, 1, 1, 0, 1]], dtype=np.uint8),
    np.array([[1, 1, 1, 0, 0], [0, 1, 1, 0, 0], [1, 1, 1, 0, 1], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0]], dtype=np.uint8),
    np.array([[1, 1, 0, 0, 0], [0, 1, 1, 1, 0], [1, 1, 1, 0, 0], [1, 0, 0, 0, 0], [1, 1, 0, 1, 0]], dtype=np.uint8),
    np.array([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0], [1, 1, 1, 1, 0], [1, 1, 0, 1, 1], [0, 1, 0, 1, 1]], dtype=np.uint8),
    np.array([[1, 1, 1, 1, 1], [0, 1, 1, 1, 1], [0, 1, 1, 0, 0], [1, 1, 1, 0, 0], [0, 1, 1, 1, 0]], dtype=np.uint8),
    np.array([[1, 1, 1, 1, 1], [0, 1, 1, 0, 0], [1, 1, 1, 1, 0], [1, 0, 1, 1, 1], [0, 1, 1, 0, 0]], dtype=np.uint8),
    np.array([[0, 1, 1, 1, 1], [1, 1, 1, 0, 0], [1, 0, 1, 0, 0], [1, 1, 1, 1, 0], [0, 0, 0, 1, 0]], dtype=np.uint8),
]


class GoPosition:
    def __init__(self, size=19):
        self.last_color = WHITE

        self.stones = np.array([[0 for y in range(size)]
                                for x in range(size)])


class AnalyzedPosition(GoPosition):

    def __init__(self, pad_size=19):
        super().__init__(size=pad_size)
        self.pad_size = pad_size
        self.cur_id = None
        self.shape = (pad_size, pad_size)
        self.num_intersections = pad_size * pad_size
        self.ownership = np.array([[0 for y in range(self.pad_size)]
                                   for x in range(self.pad_size)])
        self.continous_ownership = np.array([[0.0 for y in range(self.pad_size)]
                                             for x in range(self.pad_size)], dtype=float)
        self.final_ownership = np.array([[0 for y in range(self.pad_size)]
                                         for x in range(self.pad_size)])
        self.reset_predictions()
        self.segmentation = np.array([[0 for y in range(self.pad_size)]
                                      for x in range(self.pad_size)])
        self.computed_segmentation = False
        self.fetched_final_ownership = False

        self.board_mask = np.array([[0 for y in range(self.pad_size)]
                                    for x in range(self.pad_size)])
        self.agent = None

        self.b_score: float = 0.0
        self.w_score: float = 0.0
        self.size_x: int = 19
        self.size_y: int = 19
        self.game: KaTrainSGF = None
        self.fixed_mask = False
        self.next_move = None
        self.local_mask = None

    @classmethod
    def from_jax(cls, jax_data):  # TrainingOwnershipExample):
        obj = cls()
        obj.fixed_mask = True
        obj.stones = np.array(jax_data.state[..., 7])
        obj.ownership = np.array(jax_data.value.reshape(obj.shape))
        obj.local_mask = np.array(jax_data.mask)
        obj.board_mask = np.array(jax_data.board_mask)
        # next_move = np.array(jax_data.move)
        """
        if sum(next_move[:obj.num_intersections]) > 0:
            color = 1
        elif sum(next_move[obj.num_intersections: 2 * obj.num_intersections]) > 0:
            color = -1
        else:
            intersection = None
            color = 0
        
        if sum(next_move[:2 * obj.num_intersections]) > 0:
            move_mask = (next_move[:obj.num_intersections] + next_move[
                                                             obj.num_intersections: 2 * obj.num_intersections]).reshape(
                obj.shape)
            x = move_mask.argmax(axis=0).sum()
            y = move_mask.argmax(axis=1).sum()
            intersection = (x, y)
        """
        obj.next_move = tuple(np.array(jax_data.next_move_coords)), np.array(jax_data.next_move_color), None

        return obj

    @classmethod
    def from_gtp_log(cls, gtp_data):
        obj = cls()
        gtp_position, gtp_final_position, obj.sgf, obj.cur_id = obj.gtp_data_parse(gtp_data)
        obj.w_score = float(gtp_position['w_score'])
        obj.b_score = float(gtp_position['b_score'])
        size = gtp_position['size']
        if isinstance(size, list):
            obj.size_x, obj.size_y = size
        else:
            obj.size_x, obj.size_y = size, size

        obj.game = KaTrainSGF(obj.sgf)
        obj.board_mask[:obj.size_x, :obj.size_y] = 1

        obj.stones_from_gtp(gtp_position['stones'])
        # self.moves_from_gtp(gtp_position['moves'])
        obj.update_final_ownership(gtp_final_position)
        obj.update_ownership(gtp_position['w_own'], gtp_position['b_own'], threshold=ownership_strict_threshold)
        obj.board_segmentation()
        return obj

    @staticmethod
    def gtp_data_parse(gtp_data):
        try:
            sgf = gtp_data.pop('sgf')
            cur_id = gtp_data.pop('id')
            final = gtp_data.pop(f'{cur_id}M-last-')
            for key in gtp_data:
                cur_pos = gtp_data.pop(key)
                cur_id = key
                break
            else:
                raise KeyError("No key denoting the endgame position")
        except KeyError as ke:
            print(gtp_data)
            raise ke
        return cur_pos, final, sgf, cur_id

    def load_agent(self, agent):
        self.agent = agent

    def analyze_pos(self, mask):
        state = jnp.array(self.stones)
        state = stack_last_state(state)
        board_mask = jnp.array(self.board_mask)
        mask = jnp.array(mask)
        action_logits, ownership_map = self.agent((state, mask, board_mask), batched=False)
        self.predicted_ownership = np.array(ownership_map).reshape(self.shape)
        action_logits = jax.nn.softmax(action_logits, axis=-1)
        self.predicted_black_next_moves = np.array(action_logits[:self.num_intersections]).reshape(self.shape)
        self.predicted_white_next_moves = np.array(
            action_logits[self.num_intersections:2 * self.num_intersections]).reshape(self.shape)

    def stones_from_gtp(self, stones):
        for color in 'BW':
            for coords in stones[color]:
                stone = Move.from_gtp(coords, color)
                self.stones[stone.coords[0]][stone.coords[1]] = BLACK if stone.player == 'B' else WHITE

    def moves_from_gtp(self, moves):
        for move in moves:
            stone = Move.from_gtp(move[1], move[0])
            self.stones[stone.coords[0]][stone.coords[1]] = stone.player

    def update_ownership(self, w_ownership, b_ownership, threshold=ownership_strict_threshold):
        # The ownership map is given in the "reading" order:
        # Firstly, from right to left, then, from top to bottom
        # The GTP coordinates, however, represent the position from bottom to top (19th line is on the top)
        # Thus, we need to take self.size_y - y - 1 as a vertical index
        w_ownership = np.moveaxis(np.array(w_ownership).reshape((self.size_x, self.size_y)), 0, -1)[:, ::-1]
        b_ownership = np.moveaxis(np.array(b_ownership).reshape((self.size_x, self.size_y)), 0, -1)[:, ::-1]
        self.continous_ownership[:self.size_x, :self.size_y] = self.mean_scaled_ownership(b_ownership, w_ownership)

        self.ownership[:self.size_x, :self.size_y] = np.logical_and(
                                                      np.logical_and((w_ownership > max_ownership - threshold),
                                                      (b_ownership > max_ownership - threshold)),
                                                       self.final_ownership[:self.size_x, :self.size_y] == 1).astype(np.int8) - \
                                                     np.logical_and(
                                                      np.logical_and((w_ownership < threshold),
                                                      (b_ownership < threshold)),
                                                       self.final_ownership[:self.size_x, :self.size_y] == -1).astype(np.int8)

    def update_final_ownership(self, gtp_position, threshold=ownership_lenient_threshold):
        try:
            ownership = gtp_position['b_own']
            if ownership is None:
                ownership = gtp_position['w_own']
        except KeyError:
            ownership = gtp_position['w_own']
        if not ownership:
            return
        ownership = np.moveaxis(np.array(ownership).reshape((self.size_x, self.size_y)), 0, -1)[:, ::-1]
        self.final_ownership[:self.size_x, :self.size_y] = ((ownership > max_ownership - threshold).astype(np.int8) - (ownership < threshold).astype(np.int8))
        self.fetched_final_ownership = True

    @staticmethod
    def get_mask_dims(mask):
        if mask.sum() == 0:
            return None
        hor_mask = np.sum(mask, axis=1) > 0
        ver_mask = np.sum(mask, axis=0) > 0
        width = len(hor_mask) - np.argmax(hor_mask[::-1]) - np.argmax(hor_mask)
        height = len(ver_mask) - np.argmax(ver_mask[::-1]) - np.argmax(ver_mask)
        return width, height

    @staticmethod
    def stones_to_pos(katrain_stones, padded_shape):
        pos = np.zeros(padded_shape, dtype=np.int8)
        for st in katrain_stones:
            pos[st.coords[0]][st.coords[1]] = 1 if st.player == 'B' else -1
        return pos

    def get_single_local_pos(self, move_num=None):
        to_return = []

        num_segments = self.segmentation.max(initial=None)

        reasonable_segmentation = self.segmentation.copy()
        available_values = set(list(range(1, num_segments + 1)))
        for i in range(1, num_segments + 1):
            segment_size = (reasonable_segmentation == i).sum()
            if segment_size > 15 or segment_size < 3:
                reasonable_segmentation[reasonable_segmentation == i] = 0
                available_values.remove(i)

        if move_num is None:
            move_num = self.move_num
        node = self.game.root
        for _ in range(move_num - 1):
            node = node.children[0]
        game = BaseGame(node)
        prev_pos = np.array(self.stones_to_pos(game.stones, self.shape))
        node = node.children[0]
        game.play(node.move)
        color_to_play = -1 if node.move.player == 'B' else 1
        cur_pos = np.array(self.stones_to_pos(game.stones, self.shape))
        assert np.array_equal(cur_pos, self.stones), f"Positions don\'t match for {self.cur_id}"
        move_array = (self.stones_to_pos([node.move], self.shape) != 0).astype(np.uint8)
        dilated_move_array = cv2.dilate(move_array, kernel_small)
        max_val = (dilated_move_array * reasonable_segmentation).max()
        if max_val > 0:
            local_mask = cv2.dilate((reasonable_segmentation == max_val).astype(np.uint8), kernel_medium)
            available_values.remove(max_val)
        else:
            local_mask = cv2.dilate(move_array.astype(np.uint8), random.choice(artificial_kernels))
            local_mask = local_mask - np.logical_and(local_mask, self.segmentation)

        local_mask = np.logical_and(local_mask, self.board_mask)

        coords, player, _ = self.get_first_local_move(local_mask, node=node.children[0])

        # if random.choice([0, 1]):
        #     prev_pos = prev_pos[:, ::-1]
        #     cur_pos = cur_pos[:, ::-1]
        #     reasonable_segmentation = reasonable_segmentation[:, ::-1]
        #     local_mask = local_mask[:, ::-1]
        #     if coords:
        #         coords = (coords[0], self.pad_size - coords[1] - 1)
        # if random.choice([0, 1]):
        #     prev_pos = prev_pos[::-1, :]
        #     cur_pos = cur_pos[::-1, :]
        #     reasonable_segmentation = reasonable_segmentation[:, ::-1]
        #     local_mask = local_mask[::-1, :]
        #     if coords:
        #         coords = (self.pad_size - coords[0] - 1, coords[1])
        positions = [prev_pos * color_to_play] * 7
        positions.append(cur_pos * color_to_play)
        positions.append(np.full(self.shape, color_to_play))
        if player is not None:
            player = player * color_to_play
        to_return.append((local_mask, positions, coords, player, self.continous_ownership * color_to_play))

        same_positions = [np.array(cur_pos)] * 8
        same_positions.append(np.zeros(self.shape))
        for counter, i in enumerate(list(available_values)):
            mask = (reasonable_segmentation == i).astype(np.uint8)
            local_mask = cv2.dilate(mask, kernel_medium)
            local_mask = np.logical_and(local_mask, self.board_mask)
            coords, player, _ = self.get_first_local_move(local_mask, node=node.children[0])

            # if random.choice([0, 1]):
            #     prev_pos = prev_pos[:, ::-1]
            #     cur_pos = cur_pos[:, ::-1]
            #     reasonable_segmentation = reasonable_segmentation[:, ::-1]
            #     local_mask = local_mask[:, ::-1]
            #     if coords:
            #         coords = (coords[0], self.pad_size - coords[1] - 1)
            # if random.choice([0, 1]):
            #     prev_pos = prev_pos[::-1, :]
            #     cur_pos = cur_pos[::-1, :]
            #     reasonable_segmentation = reasonable_segmentation[:, ::-1]
            #     local_mask = local_mask[::-1, :]
            #     if coords:
            #         coords = (self.pad_size - coords[0] - 1, coords[1])

            to_return.append((local_mask, same_positions, coords, player, self.continous_ownership))
            if counter > 2:
                break
        return to_return

    @property
    def move_num(self):
        return int(self.cur_id.rsplit('M')[-1])

    def get_first_local_move(self, mask, start_move=None, node=None):
        if self.fixed_mask:
            return self.next_move
        if start_move is None:
            start_move = self.move_num + 1
        if node is None:
            node = self.game.root
            for _ in range(start_move):
                node = node.children[0]
        counter = start_move
        while True:
            if node.move and node.move.coords:
                x, y = node.move.coords
                if mask[x][y] == 1:
                    player = 1 if node.player == 'B' else -1
                    return (x, y), player, counter
            if not node.children:
                break
            counter += 1
            node = node.children[0]
        # No move was played in the masked region after starting from move start_move
        return None, None, counter

    def reset_predictions(self):
        self.predicted_ownership = np.array([[0 for y in range(self.pad_size)]
                                             for x in range(self.pad_size)])
        self.predicted_white_next_moves = np.array([[0 for y in range(self.pad_size)]
                                                    for x in range(self.pad_size)])
        self.predicted_black_next_moves = np.array([[0 for y in range(self.pad_size)]
                                                    for x in range(self.pad_size)])

    def board_segmentation(self, mode='cv'):
        #start_time = time.time()
        if mode == "manual":
            counter = 1
            for x in range(self.size_x):
                for y in range(self.size_y):
                    if self.ownership[x][y] != 0 or self.segmentation[x][y] != 0:
                        continue
                    else:
                        cur_mask = self.get_local_pos_mask((x, y), force_start=False)
                        self.segmentation[cur_mask == 1] = counter
                        counter += 1
        ownership_to_use = np.logical_and((self.ownership == 0), (self.board_mask != 0))
        ownership_to_use = ownership_to_use.astype(np.uint8)
        if mode == "scipy":
            structure = np.ones((3, 3), dtype=np.int)
            labeled, self.segmentation = label(ownership_to_use, structure)

        elif mode == "cv":
            #ownership_to_use = cv2.dilate(ownership_to_use, kernel_small)
            num_labels, self.segmentation = cv2.connectedComponents(ownership_to_use)
        #end_time = time.time()
        #print(f'Segmentation took {end_time - start_time} ms')

    # deprecated
    def get_local_pos_mask(self, intersection, force_start=False, use_precomputed=True):

        def neighbours(x, y):
            deltas = [(0, 1), (1, 0), (1, 1), (0, -1), (-1, 0), (-1, -1), (1, -1), (-1, 1),
                      (2, 0), (0, 2), (-2, 0), (0, -2)]
            nbs = [(x + delta[0], y + delta[1])
                   for delta in deltas
                   if 0 <= x + delta[0] <= self.size_x - 1 and 0 <= y + delta[1] <= self.size_y - 1]
            return nbs

        x, y = intersection

        if use_precomputed and self.computed_segmentation:
            return np.array(self.segmentation == self.segmentation[x][y])

        visited_nbs = [[0 for y in range(self.pad_size)]
                       for x in range(self.pad_size)]
        local_pos_mask = [[0 for y in range(self.pad_size)]
                          for x in range(self.pad_size)]
        visited_nbs[x][y] = 1
        if self.ownership[x][y] == 0:
            local_pos_mask[x][y] = 1

        def populate_mask(x, y, force_start=force_start):
            if self.ownership[x][y] != 0 and not force_start:
                return
            nbs = neighbours(x, y)
            for nb in nbs:
                xnb, ynb = nb
                if visited_nbs[xnb][ynb] == 1:
                    continue
                visited_nbs[xnb][ynb] = 1
                if self.ownership[xnb][ynb] == 0:
                    local_pos_mask[xnb][ynb] = 1
                populate_mask(xnb, ynb, force_start=False)

        populate_mask(x, y, force_start=force_start)
        return np.array(local_pos_mask)

    @staticmethod
    def mean_scaled_ownership(a, b):
        return AnalyzedPosition.scale_ownership((a + b) / 2)

    @staticmethod
    def scale_ownership(a):
        return (2 * a - max_ownership) / max_ownership
