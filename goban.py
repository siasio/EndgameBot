import cloudpickle
from PyQt5 import QtGui
from PyQt5.QtCore import Qt, QPoint, QRectF, QPointF, QSettings
from PyQt5.QtGui import QColor, QPainter, QPen, QBrush, QFont, QPalette, QKeySequence, QPolygonF
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QHBoxLayout, QAction, QShortcut, QVBoxLayout, \
    QPushButton, QFileDialog, QCheckBox, QRadioButton, QButtonGroup, QToolButton, QFrame
from sgf_parser import Move
import json
import os
from local_pos_masks import AnalyzedPosition
import pickle
from policies.resnet_policy import TransferResnet, ResnetPolicyValueNet128
import numpy as np

from build_tree import PositionTree, LocalPositionNode

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
        self.position_tree = PositionTree.from_a0pos(AnalyzedPosition())
        self.reset_numbers()
        self.show_segmentation = False
        self.show_ownership = False
        self.show_predicted_ownership = False
        self.show_predicted_moves = False
        self.show_black_scores = False
        self.show_white_scores = False
        self.show_actual_move = False
        self.local_mask = [[1 for y in range(self.a0pos.pad_size)]
                           for x in range(self.a0pos.pad_size)]
        self.agent = self.load_agent(os.path.join(os.getcwd(), 'a0-jax', "trained.ckpt"))
        self.a0pos.load_agent(self.agent)
        self.from_pkl = False
        self.default_event_handler = self.handle_default_event
        self.stone_numbers = np.array([[None for y in range(self.size_y)] for x in range(self.size_x)])
        self.last_number = 0

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        def draw_text(x_coord, y_coord, text_to_draw):
            rect = painter.fontMetrics().boundingRect(text_to_draw)
            cx = x_coord - rect.width() / 2
            cy = y_coord + font.pointSizeF() / 2
            painter.drawText(QPointF(cx, cy), text_to_draw)

        def should_show_percentage(i_coord, j_coord):
            return not self.ownership_only_for_mask or (self.local_mask is not None and self.local_mask[i_coord][j_coord])

        def should_show_score(i_coord, j_coord):
            return not self.scores_only_for_mask or (self.local_mask is not None and self.local_mask[i_coord][j_coord])

        def ownership_to_rgb(ownership):
            return int(255 * (1 - (ownership + 1) / 2))

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
        painter.setFont(font)
        top_black_move = np.unravel_index(np.argmax(self.a0pos.predicted_black_next_moves, axis=None),
                                          self.a0pos.predicted_black_next_moves.shape)
        top_white_move = np.unravel_index(np.argmax(self.a0pos.predicted_white_next_moves, axis=None),
                                          self.a0pos.predicted_white_next_moves.shape)
        for i in range(self.size_x):
            for j in range(self.size_y):
                x = self.margin + i * self.stone_radius * 2
                y = self.margin + j * self.stone_radius * 2

                if self.a0pos.stones[i][j]:
                    # Pen is for the border of the stone so the color should be swapped.
                    pen = QPen(QColor("white"), self.w_border, Qt.SolidLine) if self.a0pos.stones[i][
                                                                                    j] == BLACK else QPen(
                        QColor("black"), self.b_border, Qt.SolidLine)
                    painter.setPen(pen)

                    color = QColor("black" if self.a0pos.stones[i][j] == BLACK else "white")

                    painter.setBrush(QBrush(color))
                    radius_to_use = self.stone_radius * self.b_radius if self.a0pos.stones[i][
                                                                             j] == BLACK else self.stone_radius * self.w_radius
                    painter.drawEllipse(QPoint(x, y), radius_to_use, radius_to_use)

                    corresponding_number = self.stone_numbers[i][j]
                    if corresponding_number:
                        painter.setPen(QColor("white" if self.a0pos.stones[i][j] == BLACK else "black"))
                        draw_text(x, y, str(corresponding_number))

                if self.a0pos.last_move is not None:
                    if i == self.a0pos.last_move[0] and j == self.a0pos.last_move[1]:
                        #
                        # pen = QPen(QColor("white"), self.w_border, Qt.SolidLine) if self.a0pos.stones[i][
                        #                                                                 j] == BLACK else QPen(
                        #     QColor("black"), self.b_border, Qt.SolidLine)
                        # painter.setPen(pen)
                        painter.setBrush(QBrush(QColor("red")))
                        radius_to_use = self.stone_radius * self.b_radius * 0.3
                        painter.drawEllipse(QPoint(x, y), radius_to_use, radius_to_use)

                if self.show_gt_ownership and self.a0pos.ownership[i][j]:
                    rgb = ownership_to_rgb(self.a0pos.ownership[i][j])
                    painter.setBrush(QBrush(QColor(rgb, rgb, rgb, 127)))
                    painter.setPen(Qt.NoPen)
                    painter.drawRect(x, y, int(self.stone_radius), int(self.stone_radius))

                if self.ownership_choice == "show_with_color" and should_show_percentage(i, j):
                    rgb = ownership_to_rgb(self.a0pos.predicted_ownership[i][j])
                    painter.setBrush(QBrush(QColor(rgb, rgb, rgb, 127)))
                    painter.setPen(Qt.NoPen)
                    painter.drawRect(x - int(self.stone_radius // 2), y - int(int(self.stone_radius) // 2),
                                     int(self.stone_radius), int(self.stone_radius))
                elif self.ownership_choice == "show_as_percentages" and should_show_percentage(i, j):
                    painter.setPen(QColor("white" if self.a0pos.stones[i][j] == BLACK else "black"))
                    percentage = int(50 * (self.a0pos.predicted_ownership[i][j] + 1))
                    draw_text(x, y, str(percentage))

                move_size = int(self.stone_radius / 2)
                if self.move_choice == "show_top_b_w":
                    threshold = 3 * 255  # What should it be? it was written 100

                    black_move_alpha = int(100 * 255 * self.a0pos.predicted_black_next_moves[i][j])

                    if black_move_alpha > threshold and top_black_move[0] == i and top_black_move[1] == j:
                        painter.setBrush(QBrush(QColor(0, 0, 0, black_move_alpha)))
                        painter.setPen(QPen(QColor("black")))
                        painter.drawPolygon(self.get_cross(move_size, x, y))

                    white_move_alpha = int(100 * 255 * self.a0pos.predicted_white_next_moves[i][j])

                    if white_move_alpha > threshold and top_white_move[0] == i and top_white_move[1] == j:
                        painter.setBrush(QBrush(QColor(255, 255, 255, white_move_alpha)))
                        painter.setPen(QPen(QColor("white")))
                        painter.drawEllipse(x - move_size, y - move_size, 2 * move_size, 2 * move_size)

                elif self.move_choice == "black_scores" and should_show_score(i, j):
                    painter.setPen(QColor("white" if self.a0pos.stones[i][j] == BLACK else "black"))
                    num = int(100 * self.a0pos.predicted_black_next_moves[i][j])
                    draw_text(x, y, str(num))

                elif self.move_choice == "white_scores" and should_show_score(i, j):
                    painter.setPen(QColor("white" if self.a0pos.stones[i][j] == BLACK else "black"))
                    num = int(100 * self.a0pos.predicted_white_next_moves[i][j])
                    draw_text(x, y, str(num))

                if self.show_actual_move and self.local_mask is not None:
                    coords, color, _ = self.a0pos.get_first_local_move(self.local_mask)
                    if coords is not None and i == coords[0] and j == coords[1]:
                        painter.setBrush(Qt.NoBrush)
                        painter.setPen(QPen(QColor("blue"), 3, Qt.DotLine))
                        if color == 1:
                            painter.drawPolygon(self.get_cross(move_size, x, y))
                        else:
                            painter.drawEllipse(x - move_size, y - move_size, 2 * move_size, 2 * move_size)

                # if self.show_segmentation:
                #     color = colors[int(self.a0pos.segmentation[i][j])]
                #     painter.setBrush(QBrush(color))
                #     painter.setPen(QPen(color, 0, Qt.SolidLine))
                #     painter.drawRect(x - int(self.stone_radius), y - int(self.stone_radius),
                #                      2 * int(self.stone_radius), 2 * int(self.stone_radius))
                #* self.b_radius if self.a0pos.stones[i][
                #                      j] == BLACK else self.stone_radius * self.w_radius
                if self.local_mask is not None:
                    if not self.local_mask[i][j]:
                        color = QColor(255, 255, 255, 127)
                        painter.setBrush(QBrush(color))
                        painter.setPen(Qt.NoPen)  # QPen(color, 0, Qt.SolidLine))
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

    def remove_stone(self, sgf_coords):
        for prop in ["AB", "AW"]:
            placements = self.position_tree.current_node.get_list_property(prop, [])
            if sgf_coords in placements:
                self.position_tree.current_node.set_property(prop, [xy for xy in placements if xy != sgf_coords])

    def add_black_stones(self, event):
        color = "W" if event.button() == Qt.RightButton else "B"
        self.add_stone(event, color)

    def add_white_stones(self, event):
        color = "B" if event.button() == Qt.RightButton else "W"
        self.add_stone(event, color)

    def add_stone(self, event, color):
        if self.position_tree.current_node.parent is not None: # not np.all(np.array(self.stone_numbers) == None):
            print("Already has moves, cannot add stones anymore")
            return
        pos = event.pos()
        row, col = self.pixel_to_board(pos.x(), pos.y())
        prop = "A" + color
        if 0 <= row < self.size_x and 0 <= col < self.size_y:
            added_stone = Move((row, col)).sgf([19, 19])
            if not self.a0pos.stones[row][col]:
                # Place a stone at the clicked position
                try:
                    self.position_tree.current_node.add_list_property(prop, [added_stone])
                except KeyError:
                    self.position_tree.current_node.set_property(prop, [added_stone])
            else:
                self.remove_stone(added_stone)
            self.position_tree._calculate_groups()
            self.position_tree.update_a0pos_state()
            self.update()

    def add_mask(self, event):
        pos = event.pos()
        row, col = self.pixel_to_board(pos.x(), pos.y())
        if 0 <= row < self.size_x and 0 <= col < self.size_y:
            self.local_mask[row][col] = 1 - self.local_mask[row][col]
            self.update()

    def add_moves(self, event):
        color = self.last_color if event.button() == Qt.RightButton else - self.last_color
        self.last_color = color
        color = "B" if color == 1 else "W"
        pos = event.pos()
        row, col = self.pixel_to_board(pos.x(), pos.y())
        if 0 <= row < self.size_x and 0 <= col < self.size_y:
            added_stone = Move((row, col), player=color)
            self.last_number += 1
            self.stone_numbers[row][col] = self.last_number
            self.position_tree.play(added_stone)
            self.update()

    def mousePressEvent(self, event):
        self.default_event_handler(event)

    def handle_default_event(self, event):
        pos = event.pos()
        row, col = self.pixel_to_board(pos.x(), pos.y())
        if 0 <= row < self.size_x and 0 <= col < self.size_y:
            try:
                if event.button() == Qt.RightButton:
                    self.local_mask = None
                    self.a0pos.reset_predictions()
                else:
                    self.local_mask = self.a0pos.get_local_pos_mask((row, col))
                    self.a0pos.analyze_pos(self.local_mask)
                self.update()
            except:
                pass

    def undo_move(self):
        if self.position_tree.current_node.parent is not None:
            self.position_tree.current_node = self.position_tree.current_node.parent
            del self.position_tree.current_node.children[0]
        self.stone_numbers = np.where(self.stone_numbers == self.last_number, None, self.stone_numbers)
        self.last_number -= 1
        self.last_color = - self.last_color
        self.update()

    def clear_board(self):
        self.reset_numbers()
        del self.position_tree
        self.position_tree = PositionTree.from_a0pos(AnalyzedPosition())
        self.update()

    def pixel_to_board(self, x, y):
        row = round((x - self.margin) / (self.stone_radius * 2))
        col = round((y - self.margin) / (self.stone_radius * 2))
        return row, col

    def reset_numbers(self):
        self.stone_numbers = np.array([[None for y in range(self.size_y)] for x in range(self.size_x)])
        self.last_number = 0

    @property
    def a0pos(self):
        return self.position_tree.current_node.a0pos

    def visualize_position(self, gtp_position):
        a0pos = AnalyzedPosition.from_gtp_log(gtp_position) if not self.from_pkl else AnalyzedPosition.from_jax(gtp_position)
        self.position_tree = PositionTree.from_a0pos(a0pos)
        self.local_mask = self.a0pos.local_mask if self.a0pos.fixed_mask else None
        self.a0pos.load_agent(self.agent)
        self.a0pos.analyze_pos(self.local_mask, self.agent)
        self.size_x, self.size_y = self.a0pos.size_x, self.a0pos.size_y
        self.reset_numbers()
        self.update()

    def load_agent(self, ckpt_path):
        backbone = ResnetPolicyValueNet128(input_dims=(9, 9, 9), num_actions=82)
        agent = TransferResnet(backbone)
        agent = agent.eval()
        with open(ckpt_path, "rb") as f:
            loaded_agent = pickle.load(f)
            if "agent" in loaded_agent:
                loaded_agent = loaded_agent["agent"]
            agent = agent.load_state_dict(loaded_agent)
        return agent


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

        self.move_buttons = QButtonGroup()
        self.move_buttons.setProperty("name", "move choice")
        self.ownership_buttons = QButtonGroup()
        self.ownership_buttons.setProperty("name", "ownership choice")

        self.select_dir_button = QPushButton("Select Directory", self)
        self.select_dir_button.clicked.connect(self.select_directory)
        self.select_dir_button.setStyleSheet("QPushButton {"
                                             "background-color: #D3D3D3;}")

        self.set_up_button = QPushButton(self)
        self.set_up_button.setCheckable(True)
        self.set_up_button.setText("Set up position")
        self.set_up_button.toggled.connect(self.set_up_position)
        self.set_up_button.setStyleSheet("QToolButton {"
                                             "background-color: #D3D3D3;}")
        self.add_black_button = QToolButton(self)
        self.add_black_button.setCheckable(True)
        self.add_black_button.toggled.connect(self.add_black_stones)
        self.add_black_button.setText("Black")
        self.add_white_button = QToolButton(self)
        self.add_white_button.setCheckable(True)
        self.add_white_button.toggled.connect(self.add_white_stones)
        self.add_white_button.setText("White")
        self.add_moves_button = QToolButton(self)
        self.add_moves_button.setCheckable(True)
        self.add_moves_button.toggled.connect(self.add_moves)
        self.add_moves_button.setText("Moves")
        self.add_mask_button = QToolButton(self)
        self.add_mask_button.setCheckable(True)
        self.add_mask_button.toggled.connect(self.add_mask)
        self.add_mask_button.setText("Mask")
        self.undo_button = QPushButton("Undo", self)
        self.undo_button.clicked.connect(self.undo_move)
        self.clear_button = QPushButton("Clear", self)
        self.clear_button.clicked.connect(self.clear_board)
        self.set_up_layout = QHBoxLayout()
        self.set_up_layout.addWidget(self.add_black_button)
        self.set_up_layout.addWidget(self.add_white_button)
        self.set_up_layout.addWidget(self.add_moves_button)
        self.set_up_layout.addWidget(self.add_mask_button)
        self.set_up_layout.addWidget(self.undo_button)
        self.set_up_layout.addWidget(self.clear_button)
        self.set_up_group = QButtonGroup()
        self.set_up_group.addButton(self.add_black_button)
        self.set_up_group.addButton(self.add_white_button)
        self.set_up_group.addButton(self.add_moves_button)
        self.set_up_group.addButton(self.add_mask_button)
        self.set_up_group.setExclusive(True)
        self.set_up_frame = QFrame()
        self.set_up_frame.setLayout(self.set_up_layout)

        buttons_layout = QVBoxLayout()
        buttons_layout.addWidget(self.select_dir_button)
        buttons_layout.addWidget(self.set_up_button)
        buttons_layout.addWidget(self.set_up_frame)
        self.set_up_frame.hide()

        buttons_layout.addStretch()

        buttons_layout.addWidget(QLabel("Ownership:"))
        self.predicted_own_button = self.create_radio_button("Show with color", buttons_layout, self.ownership_buttons)
        self.own_numbers_button = self.create_radio_button("Show as percentages", buttons_layout, self.ownership_buttons)

        self.masked_ownership_cb = self.create_checkbox("Ownership only for mask", buttons_layout)
        self.ownership_button = self.create_checkbox("Show GT ownership", buttons_layout)

        buttons_layout.addWidget((QLabel("Next move:")))
        self.predicted_moves_button = self.create_radio_button("Show top B W", buttons_layout, self.move_buttons)
        self.black_predictions_button = self.create_radio_button("Black scores", buttons_layout, self.move_buttons)
        self.white_predictions_button = self.create_radio_button("White scores", buttons_layout, self.move_buttons)

        self.masked_predictions_cb = self.create_checkbox("Scores only for mask", buttons_layout)
        self.actual_move_button = self.create_checkbox("Show actual move", buttons_layout)

        self.black_prob_label = QLabel("Black move prob:")
        self.black_prob_label.setStyleSheet("QLabel {"
                                 "font-size: 24px;"
                                 "padding: 20px;"
                                 "border-radius: 60px;"
                                 "background-color: #D3D3D3;"
                                 "}")
        self.white_prob_label = QLabel("White move prob:")
        self.white_prob_label.setStyleSheet("QLabel {"
                                 "font-size: 24px;"
                                 "padding: 20px;"
                                 "border-radius: 60px;"
                                 "background-color: #D3D3D3;"
                                 "}")
        self.no_move_prob_label = QLabel("No move prob:")
        self.no_move_prob_label.setStyleSheet("QLabel {"
                                 "font-size: 24px;"
                                 "padding: 20px;"
                                 "border-radius: 60px;"
                                 "background-color: #D3D3D3;"
                                 "}")
        buttons_layout.addWidget(self.black_prob_label)
        buttons_layout.addWidget(self.white_prob_label)
        buttons_layout.addWidget(self.no_move_prob_label)

        self.previous_button = QPushButton("Previous position", self)
        self.previous_button.clicked.connect(self.show_previous_position)
        self.previous_button.setStyleSheet("QPushButton {"
                                           "font-size: 24px;"
                                           "padding: 20px;"
                                           "border-radius: 60px;"
                                           "background-color: #D3D3D3;"
                                           "}")

        self.next_button = QPushButton("Next position", self)
        self.next_button.clicked.connect(self.show_next_position)
        self.next_button.setStyleSheet("QPushButton {"
                                       "font-size: 24px;"
                                       "padding: 20px;"
                                       "border-radius: 60px;"
                                       "background-color: #D3D3D3;"
                                       "}")

        navigation_layout = QHBoxLayout()
        navigation_layout.addWidget(self.previous_button)
        navigation_layout.addStretch()
        navigation_layout.addWidget(self.next_button)

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

    def create_checkbox(self, text, button_layout, radio_button=False, radio_button_group: QButtonGroup = None, default_value=False):
        setting_name = text.lower().replace(" ", "_")
        setattr(self.go_board, setting_name, default_value)  # Initialize setting with default value

        new_widget = QCheckBox(text, self)
        new_widget.stateChanged.connect(lambda state: self.update_setting(setting_name, state))
        new_widget.setStyleSheet("QCheckBox { background-color: #D3D3D3; }")
        button_layout.addWidget(new_widget)
        return new_widget

    def create_radio_button(self, text, button_layout, radio_button_group: QButtonGroup):
        setting_name = text.lower().replace(" ", "_")
        group_name = radio_button_group.property("name").lower().replace(" ", "_")
        setattr(self.go_board, group_name, "")  # Initialize setting with default value
        new_widget = QRadioButton(text, self)
        if radio_button_group is not None:
            radio_button_group.addButton(new_widget)
        # The lambda expression is called only for state == True:
        new_widget.clicked.connect(lambda state: self.update_setting(group_name, setting_name))
        new_widget.setStyleSheet("QRadioButton { background-color: #D3D3D3; }")
        button_layout.addWidget(new_widget)
        return new_widget

    def update_setting(self, setting_name, state):
        setattr(self.go_board, setting_name, state)
        self.go_board.update()

    def show_previous_position(self):
        if self.current_position_index > 0:
            self.current_position_index -= 1
            self.visualize_position()

    def show_next_position(self):
        if self.current_position_index < len(self.positions) - 1:
            self.current_position_index += 1
            self.visualize_position()

    def set_up_position(self):
        # self.set_up_layout.setVisible(not self.set_up_layout.isVisible())
        if self.set_up_button.isChecked():
            self.set_up_button.setText("Finish and analyze")
            self.set_up_frame.show()
        else:
            self.set_up_button.setText("Set up position")
            self.set_up_frame.hide()
            self.go_board.position_tree.update_a0pos_state()
            self.go_board.a0pos.analyze_pos(self.go_board.local_mask, self.go_board.agent)
            self.update_buttons()
            self.go_board.update()

    def add_black_stones(self):
        if self.add_black_button.isChecked():
            self.go_board.default_event_handler = self.go_board.add_black_stones

    def add_white_stones(self):
        if self.add_white_button.isChecked():
            self.go_board.default_event_handler = self.go_board.add_white_stones

    def add_moves(self):
        if self.add_moves_button.isChecked():
            self.go_board.default_event_handler = self.go_board.add_moves

    def add_mask(self):
        if self.add_mask_button.isChecked():
            self.go_board.local_mask = [[0 for y in range(self.go_board.a0pos.pad_size)]
                          for x in range(self.go_board.a0pos.pad_size)]
            self.go_board.update()
            self.go_board.default_event_handler = self.go_board.add_mask

    def undo_move(self):
        self.go_board.undo_move()
        self.go_board.update()

    def clear_board(self):
        self.go_board.clear_board()
        self.go_board.update()

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

    def update_buttons(self):
        w_res = self.go_board.a0pos.w_score
        b_res = self.go_board.a0pos.b_score
        diff = b_res - w_res
        self.label.setText(f'Difference {abs(diff):.2f}')
        black_move_prob = self.go_board.a0pos.black_move_prob
        white_move_prob = self.go_board.a0pos.white_move_prob
        no_move_prob = self.go_board.a0pos.no_move_prob
        self.black_prob_label.setText(f"Black move prob: {black_move_prob:.2f}")
        self.white_prob_label.setText(f"White move prob: {white_move_prob:.2f}")
        self.no_move_prob_label.setText(f"No move prob: {no_move_prob:.2f}")

    def visualize_position(self):
        current_position = self.positions[self.current_position_index]
        if not self.go_board.from_pkl:
            current_position = json.loads(current_position)

        self.go_board.visualize_position(current_position)
        self.update_buttons()

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
