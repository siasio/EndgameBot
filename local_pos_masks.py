from sgf_parser import Move
import numpy as np

BLACK = 1
WHITE = -1
ownership_strict_threshold = 5
ownership_lenient_threshold = 15
max_ownership = 100


class A0Position:
    def __init__(self, gtp_position, gtp_final_position, pad_size=19):
        size = gtp_position['size']
        self.pad_size = pad_size
        if isinstance(size, list):
            self.size_x, self.size_y = size
        else:
            self.size_x, self.size_y = size, size
        self.last_color = WHITE

        self.stones = [[0 for y in range(self.pad_size)]
                       for x in range(self.pad_size)]
        self.ownership = [[0 for y in range(self.pad_size)]
                          for x in range(self.pad_size)]
        self.continous_ownership = [[0 for y in range(self.pad_size)]
                                    for x in range(self.pad_size)]
        self.final_ownership = [[0 for y in range(self.pad_size)]
                                for x in range(self.pad_size)]
        self.segmentation = np.array([[0 for y in range(self.pad_size)]
                                      for x in range(self.pad_size)])
        self.computed_segmentation = False
        self.fetched_final_ownership = False

        self.stones_from_gtp(gtp_position['stones'])
        # self.moves_from_gtp(gtp_position['moves'])
        self.update_final_ownership(gtp_final_position)
        self.update_ownership(gtp_position['w_own'], gtp_position['b_own'], threshold=ownership_strict_threshold)
        self.board_segmentation()

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
        i = 0
        for y in range(self.size_y):
            for x in range(self.size_x):
                self.continous_ownership[x][self.size_y - y - 1] = self.mean_scaled_ownership(b_ownership[i], w_ownership[i])
                if w_ownership[i] > max_ownership - threshold \
                        and b_ownership[i] > max_ownership - threshold \
                        and (self.final_ownership[x][self.size_y - y - 1] == 1 or not self.fetched_final_ownership):
                    self.ownership[x][self.size_y - y - 1] = 1
                elif w_ownership[i] < threshold \
                        and b_ownership[i] < threshold \
                        and (self.final_ownership[x][self.size_y - y - 1] == -1 or not self.fetched_final_ownership):
                    self.ownership[x][self.size_y - y - 1] = -1
                i += 1

    def update_final_ownership(self, gtp_position, threshold=ownership_lenient_threshold):
        try:
            ownership = gtp_position['b_own']
            if ownership is None:
                ownership = gtp_position['w_own']
        except KeyError:
            ownership = gtp_position['w_own']
        if not ownership:
            return
        i = 0
        for y in range(self.size_y):
            for x in range(self.size_x):
                if ownership[i] > max_ownership - threshold:
                    self.final_ownership[x][self.size_y - y - 1] = 1
                elif ownership[i] < threshold:
                    self.final_ownership[x][self.size_y - y - 1] = -1
                i += 1
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

    def get_single_local_pos(self):
        num_segments = self.segmentation.max(initial=None)
        for i in range(1, num_segments + 1):
            mask = self.segmentation == i
            w, h = self.get_mask_dims(mask)
            if 1 < w < 9 and 1 < h < 9:
                return mask
        return None

    def board_segmentation(self):
        counter = 1
        for x in range(self.size_x):
            for y in range(self.size_y):
                if self.ownership[x][y] != 0 or self.segmentation[x][y] != 0:
                    continue
                else:
                    cur_mask = self.get_local_pos_mask((x, y), force_start=False)
                    self.segmentation[cur_mask == 1] = counter
                    counter += 1

    def get_local_pos_mask(self, intersection, force_start=False, use_precomputed=True):

        def neighbours(x, y):
            deltas = [(0, 1), (1, 0), (1, 1), (0, -1), (-1, 0), (-1, -1), (1, -1), (-1, 1),
                      (2, 0), (0, 2), (-2, 0), (0, -2)]
            nbs = [ (x + delta[0], y + delta[1])
                    for delta in deltas
                    if 0 <= x + delta[0] <= self.size_x - 1 and 0 <= y + delta[1] <= self.size_y - 1 ]
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
        return A0Position.scale_ownership((a + b) / 2)

    @staticmethod
    def scale_ownership(a):
        return (2 * a - max_ownership) / max_ownership
