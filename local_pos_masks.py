from sgf_parser import Move

BLACK = 1
WHITE = -1


class LogToA0:
    def __init__(self, size=19):
        if isinstance(size, list):
            self.size_x, self.size_y = size
        else:
            self.size_x, self.size_y = size, size
        self.last_color = WHITE
        self.clean_board()

    def stones_from_gtp(self, stones):
        for color in 'BW':
            for coords in stones[color]:
                stone = Move.from_gtp(coords, color)
                self.stones[stone.coords[0]][stone.coords[1]] = BLACK if stone.player == 'B' else WHITE

    def moves_from_gtp(self, moves):
        for move in moves:
            stone = Move.from_gtp(move[1], move[0])
            self.stones[stone.coords[0]][stone.coords[1]] = stone.player

    def update_ownership(self, w_ownership, b_ownership):
        i = 0
        for y in range(self.size_y):
            for x in range(self.size_x):
                if w_ownership[i] > 95 and b_ownership[i] > 95:
                    self.ownership[x][self.size_y - y - 1] = 1
                elif w_ownership[i] < 5 and b_ownership[i] < 5:
                    self.ownership[x][self.size_y - y - 1] = -1
                i += 1

    def clean_board(self):
        self.stones = [[0 for y in range(self.size_y)] for x in range(self.size_x)]
        self.ownership = [[0 for y in range(self.size_y)] for x in range(self.size_x)]

    def visualize_position(self, gtp_position):
        size = gtp_position['size']
        if isinstance(size, list):
            self.size_x, self.size_y = size
        else:
            self.size_x, self.size_y = size, size

        self.clean_board()
        self.stones_from_gtp(gtp_position['stones'])
        # self.moves_from_gtp(gtp_position['moves'])
        self.update_ownership(gtp_position['w_own'], gtp_position['b_own'])


