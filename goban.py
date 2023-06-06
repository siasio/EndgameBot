from PyQt5.QtCore import Qt, QPoint, QRectF, QPointF, QSettings
from PyQt5.QtGui import QColor, QPainter, QPen, QBrush, QFont, QPalette, QKeySequence
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QHBoxLayout, QAction, QShortcut, QVBoxLayout, \
    QPushButton, QFileDialog
from sgf_parser import Move
import json
import os

# Define the size of the Go board
BOARD_SIZE = 19

# Define the size of each intersection in pixels
INTERSECTION_SIZE = 20

# Define the size of the margin in pixels
MARGIN_SIZE = 30

BLACK = 1
WHITE = -1


class GoBoard(QWidget):
    def __init__(self, parent=None, size=19, stone_radius=20, margin=30, b_border=2, w_border=1.5, b_radius=.92,
                 w_radius=.88, **kwargs):
        super().__init__(parent, **kwargs)
        # self.size = size
        if isinstance(size, list):
            self.size_x, self.size_y = size
        else:
            self.size_x, self.size_y = size, size
        self.stone_radius = stone_radius
        self.margin = margin
        self.b_border = b_border
        self.w_border = w_border
        self.b_radius = b_radius
        self.w_radius = w_radius

        self.setMinimumSize((self.size_y - 1) * self.stone_radius * 2 + self.margin * 2,
                            (self.size_y - 1) * self.stone_radius * 2 + self.margin * 2)

        self.last_color = WHITE
        self.clean_board()

    def stones_from_gtp(self, stones):
        for color in 'BW':
            for coords in stones[color]:
                stone = Move.from_gtp(coords, color)
                self.stones[stone.coords[0]][stone.coords[1]] = BLACK if stone.player == 'B' else WHITE

    def moves_from_gtp(self, moves):
        for move in moves:
            self.last_number += 1
            stone = Move.from_gtp(move[1], move[0])
            self.stones[stone.coords[0]][stone.coords[1]] = stone.player
            self.stone_numbers[stone.coords[0]][stone.coords[1]] = self.last_number

    def update_ownership(self, w_ownership, b_ownership):
        i = 0
        for y in range(self.size_y):
            for x in range(self.size_x):
                if w_ownership[i] > 95 and b_ownership[i] > 95:
                    self.ownership[x][self.size_y - y - 1] = 1
                elif w_ownership[i] < 5 and b_ownership[i] < 5:
                    self.ownership[x][self.size_y - y - 1] = -1
                i += 1

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw the board background with margins
        board_size_x = (self.size_x - 1) * self.stone_radius * 2 + self.margin * 2
        board_size_y = (self.size_y - 1) * self.stone_radius * 2 + self.margin * 2
        board_size = max(board_size_x, board_size_y)
        painter.fillRect(0, 0, board_size, board_size, QColor("#E3B05B"))

        # Draw the lines and star points
        pen = QPen(Qt.black, 1, Qt.SolidLine)
        painter.setPen(pen)
        for i in range(self.size_x):
            x = self.margin + i * self.stone_radius * 2
            painter.drawLine(x, self.margin, x, board_size_x - self.margin)
        for i in range(self.size_y):
            y = self.margin + i * self.stone_radius * 2
            painter.drawLine(self.margin, y, board_size_y - self.margin, y)
        if self.size_x == self.size_y == 19:
            for i in [3, 9, 15]:
                for j in [3, 9, 15]:
                    star_x = self.margin + i * self.stone_radius * 2
                    star_y = self.margin + j * self.stone_radius * 2
                    painter.setBrush(QBrush(Qt.black))
                    painter.drawEllipse(QPoint(star_x, star_y), 3, 3)

        # Draw the stones and numbers
        font = QFont("Helvetica", self.stone_radius - 4)
        for i in range(self.size_x):
            for j in range(self.size_y):
                if self.stones[i][j] is not None:
                    color = QColor("black" if self.stones[i][j] == BLACK else "white")

                    pen = QPen(QColor("white"), self.w_border, Qt.SolidLine) if self.stones[i][j] == BLACK else QPen(
                        QColor("black"), self.b_border, Qt.SolidLine)
                    painter.setPen(pen)
                    painter.setBrush(QBrush(color))
                    x = self.margin + i * self.stone_radius * 2
                    y = self.margin + j * self.stone_radius * 2
                    radius_to_use = self.stone_radius * self.b_radius if self.stones[i][
                                                                             j] == BLACK else self.stone_radius * self.w_radius
                    painter.drawEllipse(QPoint(x, y), radius_to_use, radius_to_use)

                    corresponding_number = self.stone_numbers[i][j]
                    if corresponding_number:
                        painter.setFont(font)
                        painter.setPen(QColor("white" if self.stones[i][j] == BLACK else "black"))
                        text = str(corresponding_number)
                        rect = painter.fontMetrics().boundingRect(text)
                        cx = x - rect.width() / 2
                        cy = y + font.pointSizeF() / 2
                        painter.drawText(QPointF(cx, cy), text)

                if self.ownership[i][j] is not None:
                    rgb = int(255 * (1 - (self.ownership[i][j] + 1) / 2))
                    color = QColor(rgb, rgb, rgb, 127)

                    painter.setBrush(QBrush(color))
                    pen = QPen(color, 0, Qt.SolidLine)
                    painter.setPen(pen)
                    x = self.margin + i * self.stone_radius * 2
                    y = self.margin + j * self.stone_radius * 2
                    radius_to_use = self.stone_radius * self.b_radius if self.stones[i][
                                                                             j] == BLACK else self.stone_radius * self.w_radius
                    painter.drawRect(x, y, radius_to_use, radius_to_use)

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self.undo_move()
            self.update()
            return
        pos = event.pos()
        row, col = self.pixel_to_board(pos.x(), pos.y())
        if 0 <= row < self.size_x and 0 <= col < self.size_y and self.stones[row][col] is None:
            # Place a stone at the clicked position
            self.last_color = - self.last_color
            self.stones[row][col] = self.last_color
            self.last_number += 1
            self.stone_numbers[row][col] = self.last_number

            self.update()

    # def contextMenuEvent(self, event):

    # menu = QMenu(self)
    # action = QAction("Undo move", self)
    # menu.addAction(action)
    # action.triggered.connect(self.undo_move)
    # menu.exec_(self.mapToGlobal(event.pos()))

    def undo_move(self):
        for i in range(self.size_x):
            for j in range(self.size_y):
                if self.stone_numbers[i][j] == self.last_number:
                    self.stone_numbers[i][j] = None
                    self.stones[i][j] = None
        self.last_number -= 1
        self.last_color = - self.last_color
        self.update()

    def pixel_to_board(self, x, y):
        row = round((x - self.margin) / (self.stone_radius * 2))
        col = round((y - self.margin) / (self.stone_radius * 2))
        return row, col

    def clean_board(self):
        self.stones = [[None for y in range(self.size_y)] for x in range(self.size_x)]
        self.stone_numbers = [[None for y in range(self.size_y)] for x in range(self.size_x)]
        self.ownership = [[None for y in range(self.size_y)] for x in range(self.size_x)]
        self.last_number = 0

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
        self.update()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Go Board")
        self.setStyleSheet("QWidget { background-color: #B0C4DE; }")

        # get screen resolution
        screen_resolution = QApplication.desktop().screenGeometry()
        board_width = min(screen_resolution.width(), screen_resolution.height())

        intersection_size = board_width / (2 * (BOARD_SIZE - 1 + 2 * (MARGIN_SIZE / INTERSECTION_SIZE)))
        margin_size = intersection_size * MARGIN_SIZE / INTERSECTION_SIZE

        # create the widgets
        self.go_board = GoBoard(self, size=BOARD_SIZE, stone_radius=intersection_size, margin=margin_size)
        # picture_widget = QLabel()  # replace this with your own widget

        # create a layout and add the widgets to it
        # main_layout = QHBoxLayout()
        # main_layout.addWidget(picture_widget)
        # main_layout.addWidget(self.go_board)

        self.select_dir_button = QPushButton("Select Directory", self)
        # self.select_dir_button.setGeometry(10, 440, 120, 30)
        self.select_dir_button.clicked.connect(self.select_directory)
        self.select_dir_button.setStyleSheet("QPushButton {"
                                             "background-color: #D3D3D3;}")

        self.previous_button = QPushButton("Previous position", self)
        # self.good_button.setGeometry(240, 440, 120, 120)
        self.previous_button.clicked.connect(self.show_previous_position)
        self.previous_button.setStyleSheet("QPushButton {"
                                           "font-size: 24px;"
                                           "padding: 20px;"
                                           "border-radius: 60px;"
                                           "background-color: #D3D3D3;"
                                           "}")

        self.next_button = QPushButton("Next position", self)
        # self.bad_button.setGeometry(520, 440, 120, 120)
        self.next_button.clicked.connect(self.show_next_position)
        self.next_button.setStyleSheet("QPushButton {"
                                       "font-size: 24px;"
                                       "padding: 20px;"
                                       "border-radius: 60px;"
                                       "background-color: #D3D3D3;"
                                       "}")

        # video_layout = QVBoxLayout()
        # video_layout.addWidget(self.video_label)

        navigation_layout = QHBoxLayout()
        navigation_layout.addWidget(self.previous_button)
        navigation_layout.addStretch()
        navigation_layout.addWidget(self.next_button)

        buttons_layout = QVBoxLayout()
        buttons_layout.addWidget(self.select_dir_button)
        buttons_layout.addStretch()
        self.label = QLabel("B + ?")
        self.label.setStyleSheet("QLabel {"
                                 "font-size: 24px;"
                                 "padding: 20px;"
                                 "border-radius: 60px;"
                                 "background-color: #D3D3D3;"
                                 "}")
        buttons_layout.addWidget(self.label)
        buttons_layout.addStretch()
        buttons_layout.addLayout(navigation_layout)

        main_layout = QHBoxLayout()
        main_layout.addWidget(self.go_board, 3)
        main_layout.addLayout(buttons_layout, 1)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        central_widget.setLayout(main_layout)

        # create a central widget and set the layout
        central_widget = QWidget()
        central_widget.setLayout(main_layout)

        # set the central widget of the main window
        self.setCentralWidget(central_widget)

        # maximize the window by default
        self.showMaximized()

        # enable full-screen mode with F12
        shortcut = QShortcut(QKeySequence(Qt.Key_F12), self)
        shortcut.activated.connect(self.toggleFullScreen)

    def show_previous_position(self):
        if self.current_position_index > 0:
            self.current_position_index -= 1
            self.visualize_position()

    def show_next_position(self):
        if self.current_position_index < len(self.positions) - 1:
            self.current_position_index += 1
            self.visualize_position()

    def select_directory(self):
        settings = QSettings("GoBoard", "Settings")
        last_directory = settings.value("last_directory")

        dialog = QFileDialog(self, "Select Directory")
        dialog.setFileMode(QFileDialog.AnyFile)

        if last_directory:
            dialog.setDirectory(last_directory)

        if dialog.exec_():
            selected_directory = dialog.selectedFiles()[0]
            self.selected_sgf = selected_directory
            with open(self.selected_sgf, 'r') as f:
                self.positions = f.read().splitlines()
                self.current_position_index = 0
                self.visualize_position()

            settings.setValue("last_directory", selected_directory)
            # self.update_pinned_directories(selected_directory)

            # self.video_files = sorted([f for f in os.listdir(self.video_dir) if f.endswith('.mp4')])
            # self.show_next_video()

    def visualize_position(self):
        current_position = json.loads(self.positions[self.current_position_index])
        self.go_board.visualize_position(current_position)
        w_res = current_position['w_score']
        b_res = current_position['b_score']
        diff = b_res - w_res
        self.label.setText(f'Difference {abs(diff):.2f}')

    def toggleFullScreen(self):
        if self.windowState() & Qt.WindowFullScreen:
            self.showNormal()
        else:
            self.showFullScreen()


if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
