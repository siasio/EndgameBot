import cloudpickle
from PyQt5 import QtGui
from PyQt5.QtCore import Qt, QPoint, QRectF, QPointF, QSettings
from PyQt5.QtGui import QColor, QPainter, QPen, QBrush, QFont, QPalette, QKeySequence, QPolygonF
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QHBoxLayout, QAction, QShortcut, QVBoxLayout, \
    QPushButton, QFileDialog, QCheckBox
from sgf_parser import Move
import json
import os
from local_pos_masks import AnalyzedPosition
import pickle
from policies.resnet_policy import TransferResnet, ResnetPolicyValueNet128

# Define the size of the Go board
BOARD_SIZE = 19

# Define the size of each intersection in pixels
INTERSECTION_SIZE = 20

# Define the size of the margin in pixels
MARGIN_SIZE = 30

BLACK = 1
WHITE = -1
colors = [QColor(150, 150, 150, 196)]
initial_r, initial_g, initial_b = 13, 179, 61
new_r, new_g, new_b = initial_r, initial_g, initial_b
for i in range(20):
    new_r = new_r * initial_r % 255
    new_g = new_g * initial_g % 255
    new_b = new_b * initial_b % 255
    colors.append(QColor(new_r, new_g, new_b, 127))


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
        self.a0pos = AnalyzedPosition()
        self.reset_numbers()
        self.show_segmentation = False
        self.show_ownership = False
        self.show_predicted_ownership = False
        self.show_predicted_moves = False
        self.show_actual_move = False
        self.local_mask = None
        self.load_agent("/home/test/PycharmProjects/a0-jax/trained.ckpt")
        self.from_pkl = False

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
                x = self.margin + i * self.stone_radius * 2
                y = self.margin + j * self.stone_radius * 2
                if self.a0pos.stones[i][j]:
                    color = QColor("black" if self.a0pos.stones[i][j] == BLACK else "white")

                    pen = QPen(QColor("white"), self.w_border, Qt.SolidLine) if self.a0pos.stones[i][j] == BLACK else QPen(
                        QColor("black"), self.b_border, Qt.SolidLine)
                    painter.setPen(pen)
                    painter.setBrush(QBrush(color))
                    radius_to_use = self.stone_radius * self.b_radius if self.a0pos.stones[i][
                                                                             j] == BLACK else self.stone_radius * self.w_radius
                    painter.drawEllipse(QPoint(x, y), radius_to_use, radius_to_use)

                    corresponding_number = self.stone_numbers[i][j]
                    if corresponding_number:
                        painter.setFont(font)
                        painter.setPen(QColor("white" if self.a0pos.stones[i][j] == BLACK else "black"))
                        text = str(corresponding_number)
                        rect = painter.fontMetrics().boundingRect(text)
                        cx = x - rect.width() / 2
                        cy = y + font.pointSizeF() / 2
                        painter.drawText(QPointF(cx, cy), text)

                if self.show_ownership and self.a0pos.ownership[i][j]:
                    rgb = int(255 * (1 - (self.a0pos.ownership[i][j] + 1) / 2))
                    color = QColor(rgb, rgb, rgb, 127)

                    painter.setBrush(QBrush(color))
                    painter.setPen(Qt.NoPen)
                    painter.drawRect(x, y, int(self.stone_radius), int(self.stone_radius))

                if self.show_predicted_ownership and self.local_mask is not None:
                    if self.local_mask[i][j]:
                        rgb = int(255 * (1 - (self.a0pos.predicted_ownership[i][j] + 1) / 2))
                        color = QColor(rgb, rgb, rgb, 127)
                        painter.setBrush(QBrush(color))
                        painter.setPen(QPen(color, 0, Qt.SolidLine))
                        painter.drawRect(x - int(self.stone_radius // 2), y - int(int(self.stone_radius) // 2),
                                         int(self.stone_radius), int(self.stone_radius))

                move_size = int(self.stone_radius / 2)
                if self.show_predicted_moves:
                    threshold = 100

                    black_move_alpha = int(100 * 255 * self.a0pos.predicted_black_next_moves[i][j])
                    if black_move_alpha > threshold:
                        black_color = QColor(0, 0, 0, black_move_alpha)
                        painter.setBrush(QBrush(black_color))
                        painter.setPen(QPen(QColor("black")))
                        painter.drawPolygon(self.get_cross(move_size, x, y))

                    white_move_alpha = int(100 * 255 * self.a0pos.predicted_white_next_moves[i][j])

                    if white_move_alpha > threshold:
                        white_color = QColor(255, 255, 255, white_move_alpha)
                        painter.setBrush(QBrush(white_color))
                        painter.setPen(QPen(QColor("white")))
                        painter.drawEllipse(x - move_size, y - move_size, 2 * move_size, 2 * move_size)

                if self.show_actual_move and self.local_mask is not None:
                    coords, color, _ = self.a0pos.get_first_local_move(self.local_mask)
                    if coords is not None and i == coords[0] and j == coords[1]:
                        painter.setBrush(Qt.NoBrush)
                        painter.setPen(QPen(QColor("blue"), 3, Qt.DotLine))
                        if color == 1:
                            painter.drawPolygon(self.get_cross(move_size, x, y))
                        else:
                            painter.drawEllipse(x - move_size, y - move_size, 2 * move_size, 2 * move_size)

                if self.show_segmentation:
                    color = colors[int(self.a0pos.segmentation[i][j])]
                    painter.setBrush(QBrush(color))
                    painter.setPen(QPen(color, 0, Qt.SolidLine))
                    painter.drawRect(x - int(self.stone_radius), y - int(self.stone_radius),
                                     2 * int(self.stone_radius), 2 * int(self.stone_radius))
                    #* self.b_radius if self.a0pos.stones[i][
                    #                      j] == BLACK else self.stone_radius * self.w_radius
                if self.local_mask is not None:
                    if not self.local_mask[i][j]:
                        color = QColor(255, 255, 255, 196)
                        painter.setBrush(QBrush(color))
                        painter.setPen(QPen(color, 0, Qt.SolidLine))
                        painter.drawRect(x - int(self.stone_radius), y - int(self.stone_radius),
                                         2 * int(self.stone_radius), 2 * int(self.stone_radius))
        painter.end()

    @staticmethod
    def get_cross(size, x, y):
        rotation = 45

        # Calculate the vertices of the cross
        points = [
            QPoint(- size // 2, - size * 3 // 2),
            QPoint(size // 2, - size * 3 // 2),
            QPoint(size // 2, - size // 2),
            QPoint(size * 3 // 2, - size // 2),
            QPoint(size * 3 // 2, size // 2),
            QPoint(size // 2, size // 2),
            QPoint(size // 2, size * 3 // 2),
            QPoint(- size // 2, size * 3 // 2),
            QPoint(- size // 2, size // 2),
            QPoint(- size * 3 // 2, size // 2),
            QPoint(- size * 3 // 2, - size // 2),
            QPoint(- size // 2, - size // 2)
        ]

        # Create a QPolygonF object with the calculated vertices
        polygon = QPolygonF(points)

        # Rotate the polygon around its center

        transform = QtGui.QTransform()
        transform.translate(x, y)
        transform.rotate(rotation)
        return transform.map(polygon)

    def mousePressEvent2(self, event):
        if event.button() == Qt.RightButton:
            self.undo_move()
            self.update()
            return
        pos = event.pos()
        row, col = self.pixel_to_board(pos.x(), pos.y())
        if 0 <= row < self.size_x and 0 <= col < self.size_y and not self.a0pos.stones[row][col]:
            # Place a stone at the clicked position
            self.last_color = - self.last_color
            self.a0pos.stones[row][col] = self.last_color
            self.last_number += 1
            self.stone_numbers[row][col] = self.last_number

            self.update()

    def mousePressEvent(self, event):
        pos = event.pos()
        row, col = self.pixel_to_board(pos.x(), pos.y())
        if 0 <= row < self.size_x and 0 <= col < self.size_y:
            if event.button() == Qt.RightButton:
                self.local_mask = None
                self.a0pos.reset_predictions()
            else:
                self.local_mask = self.a0pos.get_local_pos_mask((row, col))
                self.a0pos.analyze_pos(self.local_mask)
            self.update()

    def undo_move(self):
        for i in range(self.size_x):
            for j in range(self.size_y):
                if self.stone_numbers[i][j] == self.last_number:
                    self.stone_numbers[i][j] = None
                    self.a0pos.stones[i][j] = None
        self.last_number -= 1
        self.last_color = - self.last_color
        self.update()

    def pixel_to_board(self, x, y):
        row = round((x - self.margin) / (self.stone_radius * 2))
        col = round((y - self.margin) / (self.stone_radius * 2))
        return row, col

    def reset_numbers(self):
        self.stone_numbers = [[None for y in range(self.size_y)] for x in range(self.size_x)]
        self.last_number = 0

    def visualize_position(self, gtp_position):
        self.a0pos = AnalyzedPosition.from_gtp_log(gtp_position) if not self.from_pkl else AnalyzedPosition.from_jax(gtp_position)
        self.local_mask = self.a0pos.local_mask if self.a0pos.fixed_mask else None
        self.a0pos.load_agent(self.agent)
        self.size_x, self.size_y = self.a0pos.size_x, self.a0pos.size_y
        self.reset_numbers()
        self.update()

    def load_agent(self, ckpt_path):
        backbone = ResnetPolicyValueNet128(input_dims=(9, 9, 9), num_actions=82)
        self.agent = TransferResnet(backbone)
        self.agent = self.agent.eval()
        with open(ckpt_path, "rb") as f:
            loaded_agent = pickle.load(f)
            if "agent" in loaded_agent:
                loaded_agent = loaded_agent["agent"]
            self.agent = self.agent.load_state_dict(loaded_agent)


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
        self.go_board = GoBoard(self, size=int(BOARD_SIZE), stone_radius=int(intersection_size), margin=int(margin_size))
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

        self.ownership_button = QCheckBox("Show ownership", self)
        self.ownership_button.stateChanged.connect(self.show_ownership)
        self.ownership_button.setStyleSheet("QCheckBox {"
                                            "background-color: #D3D3D3;}")

        self.segmentation_button = QCheckBox("Show segmentation", self)
        self.segmentation_button.stateChanged.connect(self.show_segmentation)
        self.segmentation_button.setStyleSheet("QCheckBox {"
                                               "background-color: #D3D3D3;}")

        self.predicted_own_button = QCheckBox("Show predicted ownership", self)
        self.predicted_own_button.stateChanged.connect(self.show_predicted_own)
        self.predicted_own_button.setStyleSheet("QCheckBox {"
                                               "background-color: #D3D3D3;}")

        self.predicted_moves_button = QCheckBox("Show predicted moves", self)
        self.predicted_moves_button.stateChanged.connect(self.show_predicted_moves)
        self.predicted_moves_button.setStyleSheet("QCheckBox {"
                                               "background-color: #D3D3D3;}")

        self.actual_move_button = QCheckBox("Show actual move", self)
        self.actual_move_button.stateChanged.connect(self.show_actual_move)
        self.actual_move_button.setStyleSheet("QCheckBox {"
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
        buttons_layout.addWidget(self.ownership_button)
        buttons_layout.addWidget(self.segmentation_button)
        buttons_layout.addWidget(self.predicted_own_button)
        buttons_layout.addWidget(self.predicted_moves_button)
        buttons_layout.addWidget(self.actual_move_button)
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

    def show_segmentation(self):
        self.go_board.show_segmentation = not self.go_board.show_segmentation
        self.go_board.update()

    def show_ownership(self):
        self.go_board.show_ownership = not self.go_board.show_ownership
        self.go_board.update()

    def show_predicted_own(self):
        self.go_board.show_predicted_ownership = not self.go_board.show_predicted_ownership
        self.go_board.update()

    def show_predicted_moves(self):
        self.go_board.show_predicted_moves = not self.go_board.show_predicted_moves
        self.go_board.update()

    def show_actual_move(self):
        self.go_board.show_actual_move = not self.go_board.show_actual_move
        self.go_board.update()

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
            self.current_position_index = 0
            self.selected_sgf = selected_directory
            if self.selected_sgf.endswith(".log"):
                with open(self.selected_sgf, 'r') as f:
                    self.positions = f.read().splitlines()
                    self.visualize_position()
            elif self.selected_sgf.endswith(".pkl"):
                with open(self.selected_sgf, "rb") as f:
                    self.positions = cloudpickle.load(f)
                    self.positions = self.positions[:100]
                    self.go_board.from_pkl = True
                    self.visualize_position()

            settings.setValue("last_directory", selected_directory)
            # self.update_pinned_directories(selected_directory)

            # self.video_files = sorted([f for f in os.listdir(self.video_dir) if f.endswith('.mp4')])
            # self.show_next_video()

    def visualize_position(self):
        current_position = self.positions[self.current_position_index]
        if not self.go_board.from_pkl:
            current_position = json.loads(current_position)

        self.go_board.visualize_position(current_position)
        w_res = self.go_board.a0pos.w_score
        b_res = self.go_board.a0pos.b_score
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
