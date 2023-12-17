from sgf_parser import Move
import numpy as np
import jax.numpy as jnp
from jax.nn import softmax
from game import BaseGame, KaTrainSGF
from utils import stack_last_state
import time
import random
import os
from scipy.signal import convolve2d

# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = '.79'

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


# class GoPosition:
#     def __init__(self, size=19):
#         # self.last_color = WHITE
#
#         # self.stones = np.array([[0 for y in range(size)]
#         #                         for x in range(size)])


def find_best_match(smaller_array, bigger_array):
    left_margin = smaller_array.shape[0] // 2
    right_margin = smaller_array.shape[0] - left_margin
    top_margin = smaller_array.shape[1] // 2
    bottom_margin = smaller_array.shape[1] - top_margin
    # Perform 2D convolution
    typed_array = bigger_array.astype(np.float32)
    new_array = np.zeros_like(bigger_array, dtype=np.float32)
    # padded = np.zeros((bigger_array.shape[0] + left_margin + right_margin, bigger_array.shape[1] + top_margin + bottom_margin), dtype=np.float32)
    # padded[left_margin: left_margin + bigger_array.shape[0], top_margin: top_margin + bigger_array.shape[1]] = bigger_array
    for i in range(left_margin, bigger_array.shape[0] - right_margin):
        for j in range(top_margin, bigger_array.shape[1] - bottom_margin):
            # find part of the bigger array that matches the smaller array
            new_array[i, j] = np.sum(typed_array[i - left_margin: i + right_margin, j - top_margin: j + bottom_margin] * smaller_array)
    convolution_result = new_array  # convolve2d(typed_array, smaller_array, mode='same')

    # Find the indices of the maximum value in the convolution result
    best_matches = convolution_result == np.max(convolution_result)
    # randomly pick one of the best matches
    max_index = random.choice(np.argwhere(best_matches))
    # construct a matrix of zeros with the same shape as the bigger array
    max_index_matrix = np.zeros_like(bigger_array)
    # place the smaller array at the position of the best match
    try:
        max_index_matrix[
            max_index[0] - left_margin: max_index[0] + right_margin,
            max_index[1] - top_margin: max_index[1] + bottom_margin
        ] = smaller_array
        max_index_matrix *= bigger_array
    except ValueError as e:
        print("ValueError:", e)
        raise e

    return max_index_matrix


# result = find_best_match(smaller_array, bigger_array)
# print("Best match found at position:", result)


class AnalyzedPosition:

    def __init__(self, pad_size=19):
        # super().__init__(size=pad_size)
        self.pad_size = pad_size
        self.cur_id = None
        self.shape = (pad_size, pad_size)
        self.num_intersections = pad_size * pad_size
        self.ownership = np.array([[0 for y in range(self.pad_size)]
                                   for x in range(self.pad_size)])
        self.continuous_ownership = np.array([[0.0 for y in range(self.pad_size)]
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

        self.size_x: int = 19
        self.size_y: int = 19
        self.game: KaTrainSGF = None
        self.fixed_mask = False
        self.next_move = None
        self.local_mask = np.array([[1 for y in range(self.pad_size)]
                                    for x in range(self.pad_size)])
        self.color_to_play = 1

        stones = np.array([[0 for y in range(self.pad_size)]
                           for x in range(self.pad_size)])
        # self.stones = stones
        self.stacked_pos = stack_last_state(np.array(stones))

    @property
    def stones(self):
        if self.stacked_pos[0, 0, -1].astype(int) == -1:
            return - self.stacked_pos[..., -2]
        return self.stacked_pos[..., -2]
        # if self.stacked_pos is not None and self.stacked_pos[0, 0, -1].astype(int) == -1:
        #     return - self.__stones
        # return self.__stones

    # @stones.setter
    # def stones(self, stones):
    #     self.__stones = stones

    @classmethod
    def from_jax(cls, jax_data):  # TrainingOwnershipExample):
        obj = cls()
        obj.fixed_mask = True
        stacked_pos = np.array(jax_data.state)
        obj.color_to_play = -1 if stacked_pos[0, 0, -1] == -1 else 1
        obj.stacked_pos = stacked_pos * obj.color_to_play
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
        obj.ownership, obj.continuous_ownership = obj.update_ownership(gtp_position['w_own'], gtp_position['b_own'],
                                                                       obj.final_ownership, obj.size_x, obj.size_y,
                                                                       obj.pad_size,
                                                                       threshold=ownership_strict_threshold)
        obj.board_segmentation()
        node = obj.game.root
        for i in range(obj.move_num):
            node = node.children[0]
        obj.color_to_play = 1 if node.move.player == 'W' else -1
        return obj

    @classmethod
    def from_sgf_file(cls, sgf_file_contents, mask_from_sgf=False):
        obj = cls()
        obj.game = KaTrainSGF(sgf_file_contents)
        obj.board_mask[:obj.size_x, :obj.size_y] = 1
        obj.stones_from_sgf(obj.game)
        # obj.update_final_ownership(obj.game.root)
        # obj.ownership, obj.continuous_ownership = obj.update_ownership(obj.game.root.properties['W_OWN'], obj.game.root.properties['B_OWN'], obj.final_ownership, obj.size_x, obj.size_y, obj.pad_size, threshold=ownership_strict_threshold)
        # obj.board_segmentation()
        # node = obj.game.root
        # for i in range(obj.move_num):
        #     node = node.children[0]
        obj.color_to_play = 1  # if node.move.player == 'W' else -1
        if mask_from_sgf:
            obj.mask_from_sgf(obj.game)
        return obj

    @property
    def last_move(self):
        if self.stacked_pos is None:
            return None
        if -1 < np.sum(self.stacked_pos[..., -1]) < 1:
            # The last channel in stacked_pos is an indicator of what is last move's color
            # If it's filled with zeros, it means that no information about the last move is included
            return None
        new_stones = self.stacked_pos[..., 7] ** 2 - self.stacked_pos[..., 6] ** 2

        # if not 0 < jnp.sum(new_stones) < 2:
        #     print(f"Strange new moves in position. It seems that there are {sum(new_stones)} new moves.")

        return np.unravel_index(np.argmax(new_stones, axis=None), new_stones.shape)

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

    def analyze_and_decompose(self, mask, agent=None):
        self.analyze_pos(mask, agent)
        scaled_ownership = self.predicted_ownership * 50 + 50
        # print(scaled_ownership.astype(int))
        self.ownership, _ = self.update_ownership(scaled_ownership, scaled_ownership,
                                                  np.moveaxis(np.sign(self.predicted_ownership), 0, -1)[:, ::-1],
                                                  self.size_x, self.size_y, self.pad_size, threshold=10)
        self.ownership = np.moveaxis(self.ownership[:, ::-1], 0, -1)
        self.board_segmentation(mode='manual')

    def analyze_pos(self, mask, agent=None):
        if agent is None:
            agent = self.agent
        state = jnp.array(self.stacked_pos * self.color_to_play)
        # color = self.stacked_pos[0, 0, -1]  # it can be 1, 0, or -1
        board_mask = jnp.array(self.board_mask)
        mask = jnp.array(mask)
        action_logits, ownership_map = agent((state, mask, board_mask), batched=False)
        undecided_ownership = (abs(ownership_map) * 50 + 50 < (100 - ownership_strict_threshold / 2)) * mask
        if np.max(undecided_ownership) == 0:
            action_logits[-1] = 1_000_000
        self.predicted_ownership = np.array(ownership_map)  # .reshape(self.shape)
        if self.color_to_play == -1:
            self.predicted_ownership = - self.predicted_ownership
        if action_logits.shape[-1] == 2:
            self.predicted_black_next_moves = np.array(action_logits[..., 1])
            self.predicted_white_next_moves = np.array(action_logits[..., 0])
        else:
            action_logits = softmax(action_logits, axis=-1)
            black_next_moves = np.array(action_logits[self.num_intersections:2 * self.num_intersections]).reshape(
                self.shape)
            white_next_moves = np.array(action_logits[:self.num_intersections]).reshape(self.shape)
            self.predicted_black_next_moves = black_next_moves if self.color_to_play != -1 else white_next_moves
            self.predicted_white_next_moves = white_next_moves if self.color_to_play != -1 else black_next_moves
            self.black_move_prob = np.sum(self.predicted_black_next_moves)
            self.white_move_prob = np.sum(self.predicted_white_next_moves)
            self.no_move_prob = action_logits[-1].astype(float)
            # print(self.black_move_prob, self.white_move_prob, self.no_move_prob)

    def clear_analysis(self):
        pass

    def stones_from_sgf(self, sgf):
        for color in 'BW':
            for coords in sgf.root.properties[color] + sgf.root.properties[f'A{color}']:
                stone = Move.from_sgf(coords, sgf.root.board_size, player=color)
                self.stones[stone.coords[0]][stone.coords[1]] = BLACK if stone.player == 'B' else WHITE
        self.stacked_pos = stack_last_state(np.array(self.stones))

    def mask_from_sgf(self, sgf):
        if 'TR' in sgf.root.properties:
            mask = np.array([[1 for y in range(self.pad_size)] for x in range(self.pad_size)])
            for coords in sgf.root.properties['TR']:
                stone = Move.from_sgf(coords, sgf.root.board_size)
                mask[stone.coords[0]][stone.coords[1]] = 0
        elif 'SQ' in sgf.root.properties:
            mask = np.zeros((self.pad_size,
                             self.pad_size))  # np.array([[0 for y in range(self.pad_size)] for x in range(self.pad_size)])
            for coords in sgf.root.properties['SQ']:
                stone = Move.from_sgf(coords, sgf.root.board_size)
                mask[stone.coords[0]][stone.coords[1]] = 1
        else:
            return
        self.local_mask = mask

    def stones_from_gtp(self, stones):
        for color in 'BW':
            for coords in stones[color]:
                stone = Move.from_gtp(coords, color)
                self.stones[stone.coords[0]][stone.coords[1]] = BLACK if stone.player == 'B' else WHITE
        self.stacked_pos = stack_last_state(np.array(self.stones))

    def moves_from_gtp(self, moves):
        for move in moves:
            stone = Move.from_gtp(move[1], move[0])
            self.stones[stone.coords[0]][stone.coords[1]] = stone.player

    @staticmethod
    def update_ownership(
            w_ownership,
            b_ownership,
            final_ownership: np.ndarray,
            size_x: int,
            size_y: int,
            pad_size: int,
            threshold=ownership_strict_threshold
    ):
        # The ownership map is given in the "reading" order:
        # Firstly, from right to left, then, from top to bottom
        # The GTP coordinates, however, represent the position from bottom to top (19th line is on the top)
        # Thus, we need to take self.size_y - y - 1 as a vertical index
        w_ownership = np.moveaxis(np.array(w_ownership).reshape((size_x, size_y)), 0, -1)[:, ::-1]
        b_ownership = np.moveaxis(np.array(b_ownership).reshape((size_x, size_y)), 0, -1)[:, ::-1]
        continuous_ownership = np.array([[0.0 for y in range(pad_size)]
                                         for x in range(pad_size)], dtype=float)
        ownership = np.array([[0 for y in range(pad_size)]
                              for x in range(pad_size)])
        continuous_ownership[:size_x, :size_y] = AnalyzedPosition.mean_scaled_ownership(b_ownership, w_ownership)

        ownership[:size_x, :size_y] = np.logical_and(
            np.logical_and(
                (w_ownership > max_ownership - threshold),
                (b_ownership > max_ownership - threshold)
            ),
            final_ownership[:size_x, :size_y] == 1).astype(np.int8) - np.logical_and(
            np.logical_and((w_ownership < threshold),
                           (b_ownership < threshold)
                           ),
            final_ownership[:size_x, :size_y] == -1).astype(np.int8)
        return ownership, continuous_ownership

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
        self.final_ownership[:self.size_x, :self.size_y] = (
                (ownership > max_ownership - threshold).astype(np.int8) - (ownership < threshold).astype(np.int8))
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
            try:
                pos[st.coords[0]][st.coords[1]] = 1 if st.player == 'B' else -1
            except:
                print("Stone with no coords", st)
                pass
        return pos

    def get_single_local_pos(self, move_num=None, include_previous_moves=True):
        import cv2
        to_return = []

        num_segments = self.segmentation.max(initial=None)

        reasonable_segmentation = self.segmentation.copy()
        available_values = set(list(range(1, num_segments + 1)))
        for i in range(1, num_segments + 1):
            segment_size = (reasonable_segmentation == i).sum()
            if segment_size > 15 or segment_size < 3:
                reasonable_segmentation[reasonable_segmentation == i] = 0
                available_values.remove(i)

        secure_territories = reasonable_segmentation == 0

        node = self.game.root
        for _ in range(move_num):
            node = node.children[0]
        while node.children:
            node = node.children[0]
            coords = node.move.coords if node.move else None
            if coords is not None:
                secure_territories[coords[0]][coords[1]] = False

        random_artificial_kernel = random.choice(artificial_kernels)
        secure_segment = find_best_match(random_artificial_kernel, secure_territories)

        if move_num is None:
            move_num = self.move_num
        node = self.game.root
        back_this_many_moves = 7 if include_previous_moves else 1
        for _ in range(move_num - back_this_many_moves):
            node = node.children[0]
        game = BaseGame(node)
        prev_positions = []
        for _ in range(back_this_many_moves):
            prev_positions.append(np.array(self.stones_to_pos(game.stones, self.shape)))
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

        positions = [prev_positions[0] * color_to_play] * (8 - len(prev_positions))
        positions.extend([pos * color_to_play for pos in prev_positions[1:]])
        positions.append(cur_pos * color_to_play)
        positions.append(np.full(self.shape, color_to_play))
        if player is not None:
            player = player * color_to_play
        to_return.append((local_mask, positions, coords, player, self.continuous_ownership * color_to_play))

        # To learn about secure territories
        to_return.append((secure_segment, positions, None, None, np.round(self.continuous_ownership) * color_to_play))

        same_positions = [np.array(cur_pos) * color_to_play] * 8 if not include_previous_moves else positions[:-1]
        same_positions.append(np.zeros(self.shape))
        for counter, i in enumerate(list(available_values)):
            mask = (reasonable_segmentation == i).astype(np.uint8)
            local_mask = cv2.dilate(mask, kernel_medium)
            local_mask = np.logical_and(local_mask, self.board_mask)
            coords, player, _ = self.get_first_local_move(local_mask, node=node.children[0])
            if player is not None:
                player = player * color_to_play

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

            to_return.append((local_mask, same_positions, coords, player, self.continuous_ownership * color_to_play))
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
        self.black_move_prob = 0.0
        self.white_move_prob = 0.0
        self.no_move_prob = 0.0

    def board_segmentation(self, mode='cv'):
        """
        Segments the board into connected components
        :param mode: 'cv' for OpenCV, 'scipy' for scipy.ndimage.measurements.label, 'manual' segmenting by hand
        """
        # start_time = time.time()
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
            from scipy.ndimage.measurements import label
            structure = np.ones((3, 3), dtype=np.int)
            labeled, self.segmentation = label(ownership_to_use, structure)

        elif mode == "cv":
            import cv2
            # ownership_to_use = cv2.dilate(ownership_to_use, kernel_small)
            num_labels, self.segmentation = cv2.connectedComponents(ownership_to_use)
        # end_time = time.time()
        # print(f'Segmentation took {end_time - start_time} ms')

    # deprecated
    def get_local_pos_mask(self, intersection, force_start=False, use_precomputed=True):

        def neighbours(x, y):
            deltas = [
                (0, 1), (1, 0), (0, -1), (-1, 0),
                # (1, 1), (-1, -1), (1, -1), (-1, 1),
                # (2, 0), (0, 2), (-2, 0), (0, -2)
            ]
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
