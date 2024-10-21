from __future__ import annotations

import sys
import threading
import traceback

import cloudpickle
import yaml
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtCore import Qt, QPoint, QRectF, QPointF, QSettings, QMutexLocker, QMutex, QTimer, QSizeF, QLineF
from PyQt5.QtGui import QColor, QPainter, QPen, QBrush, QFont, QKeySequence, QPolygonF
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QHBoxLayout, QShortcut, QVBoxLayout, \
    QPushButton, QFileDialog, QCheckBox, QRadioButton, QButtonGroup, QToolButton, QFrame, QGraphicsWidget, \
    QGraphicsView, \
    QStyleOptionGraphicsItem, QGraphicsItem, QSpinBox

from common.utils import get_ground_truth, almost_equal
from helpers import logging_context_manager
from sgf_utils.sgf_parser import Move
import json
import os
import pickle
from policies.resnet_policy import TransferResnet, ResnetPolicyValueNet128
import numpy as np

from game_tree.position_tree import PyQtPositionTree
from game_tree.local_position_node import LocalPositionNode, LocalPositionSGF

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


def qp(p):
    return "({}, {})".format(p.x(), p.y())


class GoBoard(QWidget):
    def __init__(self, config='a0kata_estimated.yaml', parent=None, size=19, stone_radius=20, margin=30, b_border=2, w_border=1.5, b_radius=.92,
                 w_radius=.88, use_100_scale=True, **kwargs):
        super().__init__(parent, **kwargs)
        # self.size = size
        if isinstance(size, list):
            self.size_x, self.size_y = size
        else:
            self.size_x, self.size_y = size, size
        self.mask = np.array([[1 for y in range(self.size_y)] for x in range(self.size_x)])
        self.arr = np.array([[0 for y in range(self.size_y)] for x in range(self.size_x)])
        self.stone_radius = stone_radius
        self.margin = margin
        self.b_border = b_border
        self.w_border = w_border
        self.b_radius = b_radius
        self.w_radius = w_radius
        self.use_100_scale = use_100_scale

        self.setMinimumSize((self.size_y - 1) * self.stone_radius * 2 + self.margin * 2,
                            (self.size_y - 1) * self.stone_radius * 2 + self.margin * 2)

        self.last_color = WHITE
        with open(os.path.join('analysis_config', config), 'r') as f:
            self.config = yaml.safe_load(f)
        self.position_tree = PyQtPositionTree(LocalPositionNode(), config=self.config, parent_widget=self)
        self.reset_numbers()
        self.show_actual_move = False
        self.from_pkl = False
        self.default_event_handler = self.handle_default_event
        self.stone_numbers = np.array([[None for y in range(self.size_y)] for x in range(self.size_x)])
        self.last_number = 0

        self.mutex = QMutex()
        self.paint_event_done = threading.Event()  # Initialize the event
        self.built_tree = False
        self.gt = None

    def wait_for_paint_event(self):
        with QMutexLocker(self.mutex):
            self.paint_event_done.clear()
        QTimer.singleShot(0, self.on_paint_event_finished)

    def on_paint_event_finished(self):
        with QMutexLocker(self.mutex):
            self.paint_event_done.set()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        def draw_text(x_coord, y_coord, text_to_draw):
            rect = painter.fontMetrics().boundingRect(text_to_draw)
            cx = x_coord - rect.width() / 2
            cy = y_coord + font.pointSizeF() / 2
            painter.drawText(QPointF(cx, cy), text_to_draw)

        def should_show_percentage(i_coord, j_coord):
            return not self.ownership_only_for_mask or (self.position_tree.mask is not None and self.position_tree.mask[i_coord][j_coord])

        def should_show_score(i_coord, j_coord):
            return not self.scores_only_for_mask or (self.position_tree.mask is not None and self.position_tree.mask[i_coord][j_coord])

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
        if self.built_tree:
            evaluation = self.position_tree.eval_for_node()
            arr = self.position_tree.arr
            mask = self.position_tree.mask
            top_black_move = np.unravel_index(np.argmax(evaluation.black_moves, axis=None),
                                              evaluation.black_moves.shape)
            top_white_move = np.unravel_index(np.argmax(evaluation.white_moves, axis=None),
                                              evaluation.white_moves.shape)
            gt_ownership = evaluation.ownership  # self.position_tree.current_node.calculated_ownership if self.position_tree.current_node.calculated_ownership is not None else self.a0pos.ownership
        else:
            arr = self.arr
            mask = self.mask

        for i in range(self.size_x):
            for j in range(self.size_y):
                x = self.margin + i * self.stone_radius * 2
                y = self.margin + j * self.stone_radius * 2

                if arr[i][j]:
                    # Pen is for the border of the stone so the color should be swapped.
                    pen = QPen(QColor("white"), self.w_border, Qt.SolidLine) if arr[i][
                                                                                    j] == BLACK else QPen(
                        QColor("black"), self.b_border, Qt.SolidLine)
                    painter.setPen(pen)

                    color = QColor("black" if arr[i][j] == BLACK else "white")

                    painter.setBrush(QBrush(color))
                    radius_to_use = self.stone_radius * self.b_radius if arr[i][
                                                                             j] == BLACK else self.stone_radius * self.w_radius
                    painter.drawEllipse(QPoint(x, y), radius_to_use, radius_to_use)

                    corresponding_number = self.stone_numbers[i][j]
                    if corresponding_number:
                        painter.setPen(QColor("white" if arr[i][j] == BLACK else "black"))
                        draw_text(x, y, str(corresponding_number))

                # if self.show_gt_ownership and gt_ownership[i][j]:
                #     rgb = ownership_to_rgb(gt_ownership[i][j])
                #     painter.setBrush(QBrush(QColor(rgb, rgb, rgb, 127)))
                #     painter.setPen(Qt.NoPen)
                #     painter.drawRect(x, y, int(self.stone_radius), int(self.stone_radius))

                if self.built_tree:
                    if self.position_tree.current_node.move is not None and \
                            i == self.position_tree.current_node.move.coords[0] and \
                            j == self.position_tree.current_node.move.coords[1] and self.stone_numbers[i][j] is None:
                        painter.setBrush(QBrush(QColor("red")))
                        radius_to_use = self.stone_radius * self.b_radius * 0.3
                        painter.drawEllipse(QPoint(x, y), radius_to_use, radius_to_use)
                    if self.ownership_choice == "show_with_color" and should_show_percentage(i, j):
                        rgb = ownership_to_rgb(gt_ownership[i][j])
                        painter.setBrush(QBrush(QColor(rgb, rgb, rgb, 127)))
                        painter.setPen(Qt.NoPen)
                        painter.drawRect(x - int(self.stone_radius // 2), y - int(int(self.stone_radius) // 2),
                                         int(self.stone_radius), int(self.stone_radius))
                    elif self.ownership_choice == "show_as_percentages" and should_show_percentage(i, j):
                        painter.setPen(QColor("white" if arr[i][j] == BLACK else "black"))
                        percentage = int(50 * (gt_ownership[i][j] + 1)) if self.use_100_scale else round(gt_ownership[i][j])
                        draw_text(x, y, str(percentage))
                    elif self.ownership_choice == "don't_show":
                        pass

                    move_size = int(self.stone_radius / 2)
                    if self.move_choice == "show_top_b_w":
                        threshold = 3 * 255  # What should it be? it was written 100

                        black_move_alpha = int(100 * 255 * evaluation.black_moves[i][j] * evaluation.black_prob)

                        if black_move_alpha > threshold and top_black_move[0] == i and top_black_move[1] == j:
                            painter.setBrush(QBrush(QColor(0, 0, 0, black_move_alpha)))
                            painter.setPen(QPen(QColor("black")))
                            painter.drawPolygon(self.get_cross(move_size, x, y))

                        white_move_alpha = int(100 * 255 * evaluation.white_moves[i][j] * evaluation.white_prob)

                        if white_move_alpha > threshold and top_white_move[0] == i and top_white_move[1] == j:
                            painter.setBrush(QBrush(QColor(255, 255, 255, white_move_alpha)))
                            painter.setPen(QPen(QColor("white")))
                            painter.drawEllipse(x - move_size, y - move_size, 2 * move_size, 2 * move_size)

                    elif self.move_choice == "black_scores" and should_show_score(i, j):
                        painter.setPen(QColor("white" if arr[i][j] == BLACK else "black"))
                        num = int(100 * evaluation.black_moves[i][j])
                        draw_text(x, y, str(num))

                    elif self.move_choice == "white_scores" and should_show_score(i, j):
                        painter.setPen(QColor("white" if arr[i][j] == BLACK else "black"))
                        num = int(100 * evaluation.white_moves[i][j])
                        draw_text(x, y, str(num))

                    # if self.show_actual_move and self.local_mask is not None:
                    #     coords, color, _ = self.a0pos.get_first_local_move(self.local_mask)
                    #     color_to_multiply = self.a0pos.stacked_pos[..., -1][0][0]
                    #     color = color * (-color_to_multiply) if color_to_multiply != 0 else color
                    #     if coords is not None and i == coords[0] and j == coords[1]:
                    #         painter.setBrush(Qt.NoBrush)
                    #         painter.setPen(QPen(QColor("blue"), 3, Qt.DotLine))
                    #         if color == 1:
                    #             painter.drawPolygon(self.get_cross(move_size, x, y))
                    #         else:
                    #             painter.drawEllipse(x - move_size, y - move_size, 2 * move_size, 2 * move_size)

                    # if self.show_segmentation:
                    #     color = colors[int(self.a0pos.segmentation[i][j])]
                    #     painter.setBrush(QBrush(color))
                    #     painter.setPen(QPen(color, 0, Qt.SolidLine))
                    #     painter.drawRect(x - int(self.stone_radius), y - int(self.stone_radius),
                    #                      2 * int(self.stone_radius), 2 * int(self.stone_radius))
                    #* self.b_radius if self.a0pos.stones[i][
                    #                      j] == BLACK else self.stone_radius * self.w_radius
                # if self.position_tree.mask is not None:
                if not mask[i][j]:
                    color = QColor(255, 255, 255, 127)
                    painter.setBrush(QBrush(color))
                    painter.setPen(Qt.NoPen)  # QPen(color, 0, Qt.SolidLine))
                    painter.drawRect(x - int(self.stone_radius), y - int(self.stone_radius),
                                     2 * int(self.stone_radius), 2 * int(self.stone_radius))
        # Draw GameTree widget on the right side of the board
        painter.end()
        if not self.paint_event_done:
            QTimer.singleShot(0, self.on_paint_event_finished)

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
            if not self.position_tree.arr[row][col]:
                # Place a stone at the clicked position
                try:
                    self.position_tree.current_node.add_list_property(prop, [added_stone])
                except KeyError:
                    self.position_tree.current_node.set_property(prop, [added_stone])
            else:
                self.remove_stone(added_stone)
            self.position_tree._calculate_groups()
            # self.position_tree.update_a0pos_state()
            self.update()

            self.position_tree.reset_tree()

    def add_mask(self, event):
        pos = event.pos()
        row, col = self.pixel_to_board(pos.x(), pos.y())
        if 0 <= row < self.size_x and 0 <= col < self.size_y:
            self.position_tree.mask[row][col] = 1 - self.position_tree.mask[row][col]
            self.update()

    def add_moves(self, event):
        # print(self.a0pos.stones)
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
        self.update()
        # pos = event.pos()
        # row, col = self.pixel_to_board(pos.x(), pos.y())
        # if 0 <= row < self.size_x and 0 <= col < self.size_y:
        #     try:
        #         if event.button() == Qt.RightButton:
        #             self.local_mask = None
        #             self.a0pos.reset_predictions()
        #         else:
        #             self.local_mask = self.a0pos.get_local_pos_mask((row, col))
        #             self.a0pos.analyze_pos(self.local_mask)
        #         self.update()
        #     except:
        #         pass

    def undo_move(self):
        # TODO: Currently, it's not working well
        if self.position_tree.current_node.parent is not None:
            self.position_tree.undo()
        self.stone_numbers = np.where(self.stone_numbers == self.last_number, None, self.stone_numbers)
        self.last_number -= 1
        self.last_color = - self.last_color
        self.update()

    def clear_board(self):
        self.reset_numbers()
        initialized_engines = self.position_tree.eval.evaluator_registry if self.position_tree else {}
        del self.position_tree
        self.position_tree = PyQtPositionTree(LocalPositionNode(), config=self.config, parent_widget=self, **initialized_engines)
        # self.position_tree.load_agent(self.agent)
        # self.position_tree.goban = self
        self.update()

    def pixel_to_board(self, x, y):
        row = round((x - self.margin) / (self.stone_radius * 2))
        col = round((y - self.margin) / (self.stone_radius * 2))
        return row, col

    def reset_numbers(self):
        self.stone_numbers = np.array([[None for y in range(self.size_y)] for x in range(self.size_x)])
        self.last_number = 0

    def visualize_position(self, gtp_position):
        # a0pos = AnalyzedPosition.from_gtp_log(gtp_position) if not self.from_pkl else AnalyzedPosition.from_jax(gtp_position)
        # self.position_tree = PositionTree.from_a0pos(a0pos, parent_widget=self)
        # print(gtp_position)
        # Set str(gtp_position) as the window title
        self.parent().setWindowTitle(str(gtp_position))

        assert gtp_position.endswith('.sgf'), 'Parsing of non sgf positions hasn\'t been implemented yet'
        root_node: LocalPositionNode = LocalPositionSGF.parse_file(gtp_position)
        self.gt = None
        if root_comment := root_node.get_property("C"):
            try:
                self.gt = get_ground_truth(root_comment)
            except AssertionError:
                pass
        intitialized_engines = self.position_tree.eval.evaluator_registry if self.position_tree else {}
        del self.position_tree
        self.position_tree = PyQtPositionTree(root_node, config=self.config, parent_widget=self, game_name=os.path.basename(gtp_position), **intitialized_engines)
        # self.position_tree.goban = self
        # self.position_tree.update_a0pos_state()

        # self.local_mask = self.a0pos.local_mask if self.a0pos.fixed_mask else None
        # self.position_tree.load_agent(self.agent)
        # self.a0pos.analyze_pos(self.local_mask, self.agent)
        self.size_x, self.size_y = self.position_tree.board_size
        self.arr = self.position_tree.arr
        self.mask = self.position_tree.mask
        self.reset_numbers()
        self.update()

# Write a class, inheriting from QWidget, which shows a game tree. The game tree will be a binary tree.
# The root node should be represented on the top, and its children should be represented below it.


class CurrentNodeKeeper:
    def __init__(self, current_node=None):
        self.current_node = current_node


class DrawnTree(QGraphicsView):
    def __init__(self, parent: GoBoard = None, **kwargs):
        super().__init__(parent, **kwargs)

        # self.setMouseTracking(True)
        self.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.main_scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self.main_scene)

        l = QtWidgets.QGraphicsAnchorLayout()
        w = QtWidgets.QGraphicsWidget()
        self.main_scene.sceneRectChanged.connect(self.update_widget)
        # self.horizontalScrollBar().valueChanged.connect(self.update_widget)
        # self.verticalScrollBar().valueChanged.connect(self.update_widget)
        w.setLayout(l)
        self.main_scene.addItem(w)
        self.main_widget = w
        self.main_layout = l
        self.node_dict = {}
        self.current_node_keeper = CurrentNodeKeeper(None)
        self.position_tree = None

    def create_tree(self, position_tree):
        if position_tree == self.position_tree:
            return
        self.position_tree = position_tree
        def draw_node(node, parnt=None, corner=True, anchor1=None, anchor2=None, hor_spac=0.0, node_radius=40.0, sibling_node=None, spac_from_parnt=None):
            if node not in self.node_dict:
                node_widget = NodeWidget(drawn_tree=self, position_tree=position_tree, node=node, size=QSizeF(node_radius, node_radius), current_node=self.current_node_keeper, parent_node=parnt)
                self.node_dict[node] = node_widget
                draw_line = parnt is not None

                if parnt is None:
                    parnt = self.main_layout
                    second_anchor = Qt.AnchorTop
                else:
                    second_anchor = Qt.AnchorBottom
                self.main_layout.addAnchor(
                    node_widget,
                    Qt.AnchorTop,
                    parnt,
                    second_anchor,
                )

                if sibling_node is not None:
                    assert spac_from_parnt is None
                    anch = self.main_layout.addAnchor(
                        node_widget,
                        anchor2,
                        sibling_node,
                        anchor1,
                    )
                    anch.setSpacing(hor_spac - node_radius)
                elif corner:
                    assert spac_from_parnt is not None
                    anch = self.main_layout.addAnchor(
                        node_widget,
                        anchor1,
                        parnt,
                        anchor2,
                    )
                    anch.setSpacing(spac_from_parnt)
                else:
                    self.main_layout.addAnchor(
                        node_widget,
                        Qt.AnchorHorizontalCenter,
                        parnt,
                        Qt.AnchorHorizontalCenter,
                    )
                if draw_line:
                    anchor_line = AnchorLine(node_widget, parnt)
                    self.main_scene.addItem(anchor_line)

            node_widget: NodeWidget = self.node_dict[node]
            node_widget.update()

            if node.children:
                if len(node.children) == 1:
                    # In the current version of the program we always have at least 2 children (at least 1 per color)
                    draw_node(node.children[0], node_widget, corner=False, anchor1=Qt.AnchorTop, anchor2=Qt.AnchorBottom, hor_spac=hor_spac, node_radius=node_radius)
                else:
                    new_hor_spac = hor_spac / len(node.children)
                    new_node_radius = node_radius * .8
                    children_sorted = [c for c in node.children if c.player == "B"] + [c for c in node.children if c.player == "W"]
                    sibling_node = None
                    for child in children_sorted:
                        if sibling_node is None:
                            spac_from_parnt = (hor_spac - new_hor_spac - node_radius - new_node_radius) / 2
                        else:
                            spac_from_parnt = None
                        draw_node(child, node_widget, corner=True, anchor1=Qt.AnchorRight, anchor2=Qt.AnchorLeft, hor_spac=new_hor_spac, node_radius=new_node_radius, sibling_node=sibling_node, spac_from_parnt=spac_from_parnt)
                        sibling_node = self.node_dict[child]

        node_radius = self.main_widget.geometry().width() / 15

        depth = max(position_tree.root.expanded_tree_depth, 9)
        max_ver_spac = self.main_widget.geometry().height() / depth - node_radius
        self.main_layout.setVerticalSpacing(max_ver_spac)
        self.current_node_keeper.current_node = position_tree.root
        print("About to draw the node")
        draw_node(
            position_tree.root,
            parnt=None,
            corner=False,
            anchor1=Qt.AnchorTop,
            anchor2=Qt.AnchorTop,
            hor_spac=self.main_widget.geometry().width(),  # * .5 - 2 * node_radius,
            node_radius=node_radius,
        )
        self.update()

    def clear_tree(self):
        if self.position_tree is not None:
            self.position_tree.go_to_node(self.position_tree.root)
            self.position_tree.reset_tree()
            self.position_tree = None
        # Delete all created NodeWidgets and AnchorLines from the scene
        for node in self.node_dict:
            if node.parent is not None:
                self.main_scene.removeItem(self.node_dict[node])
        for item in self.main_scene.items():
            if isinstance(item, AnchorLine):
                self.main_scene.removeItem(item)
        # update what is shown in the window
        self.update()
        # self.create_tree()

    def update_widget(self):
        vp = self.viewport().mapFromParent(QtCore.QPoint())
        tl = self.mapToScene(vp)
        geo = self.main_widget.geometry()
        geo.setTopLeft(tl)
        self.main_widget.setGeometry(0, 0, self.contentsRect().width(), self.contentsRect().height())

    def resizeEvent(self, event):
        self.update_widget()
        super().resizeEvent(event)

    # On bottom arrow call the mouse press event of the first child of the current node
    def keyPressEvent(self, event):
        current_node_widget = self.node_dict[self.current_node_keeper.current_node]
        if event.key() == Qt.Key_Down:
            if current_node_widget.children_nodes:
                # call the mousePressEvent as if the left button was pressed
                current_node_widget.children_nodes[0].mousePressEvent(QtGui.QMouseEvent(QtGui.QMouseEvent.MouseButtonPress, QtCore.QPoint(), QtCore.Qt.LeftButton, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier))
        elif event.key() == Qt.Key_Up:
            if current_node_widget.parent_node:
                current_node_widget.parent_node.mousePressEvent(QtGui.QMouseEvent(QtGui.QMouseEvent.MouseButtonPress, QtCore.QPoint(), QtCore.Qt.LeftButton, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier))
        elif event.key() == Qt.Key_Right:
            parent = current_node_widget.parent_node
            if parent:
                for child_index, child in enumerate(parent.children_nodes):
                    if child == current_node_widget:
                        if child_index + 1 < len(parent.children_nodes):
                            parent.children_nodes[child_index + 1].mousePressEvent(QtGui.QMouseEvent(QtGui.QMouseEvent.MouseButtonPress, QtCore.QPoint(), QtCore.Qt.LeftButton, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier))
                        break
                else:
                    raise ValueError("Current node is not a child of its parent!")
        elif event.key() == Qt.Key_Left:
            parent = current_node_widget.parent_node
            if parent:
                for child_index, child in enumerate(parent.children_nodes):
                    if child == current_node_widget:
                        if child_index - 1 >= 0:
                            parent.children_nodes[child_index - 1].mousePressEvent(QtGui.QMouseEvent(QtGui.QMouseEvent.MouseButtonPress, QtCore.QPoint(), QtCore.Qt.LeftButton, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier))
                        break
                else:
                    raise ValueError("Current node is not a child of its parent!")

    # def mouseMoveEvent(self, event):
    #     pos = event.pos()
    #     self.printStatus(pos)
    #     super().mouseMoveEvent(event)
    #
    # def printStatus(self, pos):
    #     pass
        # msg = "Viewport Position: " + str(qp(pos))
        # v = self.mapToScene(pos)
        # v = QtCore.QPoint(v.x(), v.y())
        # msg = msg + ",  Mapped to Scene: " + str(qp(v))
        # v = self.mapToScene(self.viewport().rect()).boundingRect()
        # msg = msg + ",  viewport Mapped to Scene: " + str(qp(v))
        # v2 = self.mapToScene(QtCore.QPoint(0, 0))
        # msg = msg + ",  (0, 0) to scene: " + qp(v2)
        # self.messageChanged.emit(msg)


def player_or_none(node):
    if "B" in node.properties:
        return "B"
    elif "W" in node.properties:
        return "W"
    else:
        return None


class AnchorLine(QGraphicsItem):
    def __init__(self, item1: QGraphicsItem, item2: QGraphicsItem):
        super().__init__()
        self.item1 = item1
        self.item2 = item2
        self.setZValue(-1)  # Ensure the line is drawn beneath the items

    def boundingRect(self):
        return QRectF()

    def paint(self,
          painter: QPainter,
          option: QStyleOptionGraphicsItem,
          widget: QWidget | None = ...) -> None:
        if self.item1.scene() is not None and self.item2.scene() is not None:
            first_pos = self.item1.scenePos()
            second_pos = self.item2.scenePos()
            line = QLineF(
                first_pos.x() + self.item1.size().width() / 2,
                first_pos.y(),
                second_pos.x() + self.item2.size().width() / 2,
                second_pos.y() + self.item1.size().height(),
            )
            painter.setPen(QPen(Qt.black, 1, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawLine(line)


class NodeWidget(QGraphicsWidget):
    def __init__(self, drawn_tree=None, position_tree: PyQtPositionTree = None, node: LocalPositionNode = None, size=QSizeF(20, 20), current_node=None, parent_node=None):
        super().__init__()
        self.setAcceptHoverEvents(True)
        self.node = node
        self.position_tree = position_tree
        self.drawn_tree = drawn_tree

        self.setMinimumSize(size)
        self.setMaximumSize(size)

        player_colors = {None: Qt.green, 'B': Qt.black, 'W': Qt.white}
        player_anticolors = {None: Qt.green, 'B': Qt.white, 'W': Qt.black}

        pl = player_or_none(node)
        self.color = player_colors[pl]
        self.anticolor = player_anticolors[pl]
        self.draw_ellipse = pl is not None

        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(str(node.move))
        self.current_node_keeper = current_node
        self.children_nodes = []
        self.parent_node = parent_node
        if parent_node is not None:
            parent_node.children_nodes.append(self)

    def paint(self,
              painter: QPainter,
              option: QStyleOptionGraphicsItem,
              widget: QWidget | None = ...) -> None:
        brush = QtGui.QBrush()
        brush.setColor(QtGui.QColor(self.color))
        brush.setStyle(Qt.SolidPattern)

        # draw a circle instead of rectangle
        painter.setBrush(brush)
        anticolor = QtGui.QColor(self.anticolor)
        pen_thickness = 1
        if self.current_node_keeper.current_node == self.node:
            anticolor = QtGui.QColor("red")
            pen_thickness = 3
        painter.setPen(QtGui.QPen(anticolor, pen_thickness, Qt.SolidLine))
        # painter.setPen(anticolor)
        if self.draw_ellipse:
            painter.drawEllipse(self.boundingRect())
        else:
            painter.drawRect(self.boundingRect())

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.position_tree.go_to_node(self.node)
            self.current_node_keeper.current_node = self.node
            self.drawn_tree.update()
        else:
            print(f'Unknown button {event.button()}')


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Go Board")
        self.setStyleSheet("QWidget { background-color: #B0C4DE; }")
        self.show_as_temp = False
        # get screen resolution
        screen_resolution = QApplication.desktop().screenGeometry()
        board_width = min(screen_resolution.width(), screen_resolution.height())

        intersection_size = board_width / (2 * (BOARD_SIZE - 1 + 2 * (MARGIN_SIZE / INTERSECTION_SIZE)))
        margin_size = intersection_size * MARGIN_SIZE / INTERSECTION_SIZE

        # create the widgets
        self.go_board = GoBoard(parent=self, size=int(BOARD_SIZE), stone_radius=int(intersection_size), margin=int(margin_size))
        self.drawn_tree = DrawnTree(self.go_board)
        self.drawn_tree.setStyleSheet("QWidget { background-color: #B19886; }")

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
        self.set_up_button.setStyleSheet("QPushButton {"
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
        self.detect_mask_button = QToolButton(self)
        self.detect_mask_button.setCheckable(True)
        self.detect_mask_button.toggled.connect(self.detect_mask)
        self.detect_mask_button.setText("Detect")
        self.undo_button = QPushButton("Undo", self)
        self.undo_button.clicked.connect(self.undo_move)
        self.clear_button = QPushButton("Clear", self)
        self.clear_button.clicked.connect(self.clear_board)
        self.set_up_layout = QHBoxLayout()
        self.set_up_layout.addWidget(self.add_black_button)
        self.set_up_layout.addWidget(self.add_white_button)
        self.set_up_layout.addWidget(self.add_moves_button)
        self.set_up_layout.addWidget(self.add_mask_button)
        self.set_up_layout.addWidget(self.detect_mask_button)
        self.set_up_layout.addWidget(self.undo_button)
        self.set_up_layout.addWidget(self.clear_button)
        self.set_up_group = QButtonGroup()
        self.set_up_group.addButton(self.add_black_button)
        self.set_up_group.addButton(self.add_white_button)
        self.set_up_group.addButton(self.add_moves_button)
        self.set_up_group.addButton(self.add_mask_button)
        self.set_up_group.setExclusive(True)
        # Add description of the slider
        self.depth_label = QLabel("Depth of the game tree:")

        self.depth_spinbox = QSpinBox(self)
        # self.frames_spinbox.setGeometry(140, 480, 100, 30)
        self.depth_spinbox.setMinimum(1)  # Set the minimum value for the spinbox
        self.depth_spinbox.setMaximum(8)  # Set the maximum value for the spinbox
        self.depth_spinbox.setValue(6)
        self.depth_spinbox.setStyleSheet("QPushButton {"
                                          "font-size: 24px;"
                                          "padding: 20px;"
                                          "border-radius: 60px;"
                                          "}")
        # Add a QSlider to set the depth of the game tree (from 0 to 8)
        # self.depth_slider = QSlider(Qt.Horizontal)
        # self.depth_slider.setMinimum(0)
        # self.depth_slider.setMaximum(8)
        # self.depth_slider.setValue(0)
        # self.depth_slider.setTickPosition(QSlider.TicksBelow)
        # self.depth_slider.setTickInterval(1)
        # # Add tick labels for every depth displayed as numbers
        # self.depth_slider.setTickPosition(QSlider.TicksBothSides)
        #
        # self.depth_slider.valueChanged.connect(self.update_depth)
        self.set_up_general_layout = QVBoxLayout()
        self.set_up_general_layout.addLayout(self.set_up_layout)
        self.set_up_general_layout.addWidget(self.depth_label)
        self.set_up_general_layout.addWidget(self.depth_spinbox)
        self.set_up_frame = QFrame()
        self.set_up_frame.setLayout(self.set_up_general_layout)

        buttons_layout = QVBoxLayout()
        buttons_layout.addWidget(self.select_dir_button)
        buttons_layout.addWidget(self.set_up_button)
        buttons_layout.addWidget(self.set_up_frame)
        self.set_up_frame.hide()
        # self.depth_slider.hide()

        # self.calculate_button = QPushButton("Calculate temperature", self)
        # self.calculate_button.clicked.connect(self.calculate_score)
        # self.calculate_button.setStyleSheet("QPushButton {"
        #                                      "background-color: #D3D3D3;}")
        # self.calculate_button.hide()
        # buttons_layout.addWidget(self.calculate_button)
        # buttons_layout.addWidget(self.depth_slider)

        buttons_layout.addStretch()

        buttons_layout.addWidget(QLabel("Ownership:"))
        self.predicted_own_button = self.create_radio_button("Show with color", buttons_layout, self.ownership_buttons)
        self.own_numbers_button = self.create_radio_button("Show as percentages", buttons_layout, self.ownership_buttons)
        self.no_own_button = self.create_radio_button("Don't show", buttons_layout, self.ownership_buttons)

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
        self.label = QLabel("Game tree not built yet")
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
        main_layout.addWidget(self.drawn_tree, 1)
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

    # def update_depth(self):
    #     self.go_board.position_tree.max_depth = self.depth_slider.value()
    #     self.go_board.update()

    def show_previous_position(self):
        if self.current_position_index > 0:
            self.current_position_index -= 1
            self.visualize_position()

    def show_next_position(self):
        if self.current_position_index < len(self.positions) - 1:
            self.current_position_index += 1
            self.visualize_position()
            self.set_up_position()
            self.set_up_position()

    def set_up_position(self):
        if self.set_up_button.isChecked():
            self.drawn_tree.clear_tree()
            self.set_up_button.setText("Finish and analyze")
            self.set_up_frame.show()
            # self.depth_slider.show()
            self.set_up_group.buttons()[0].setChecked(True)
            self.update_buttons()
        else:
            self.set_up_button.setText("Set up position")
            self.set_up_frame.hide()
            # self.depth_slider.hide()
            self.go_board.position_tree.max_depth = self.depth_spinbox.value()
            # self.go_board.position_tree.update_a0pos_state()
            # self.go_board.a0pos.local_mask = self.go_board.local_mask
            # self.go_board.a0pos.analyze_pos(self.go_board.local_mask, self.go_board.agent)
            self.update_buttons()
            self.go_board.update()
            self.calculate_score()

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
            if np.min(self.go_board.position_tree.mask) > 0:
                self.go_board.position_tree.mask = [[0 for y in range(19)]
                              for x in range(19)]
            self.go_board.update()
            self.go_board.default_event_handler = self.go_board.add_mask

    def detect_mask(self):
        raise NotImplementedError
        # if self.detect_mask_button.isChecked():
        #     self.go_board.local_mask = [[1 for y in range(19)]
        #                   for x in range(19)]
        #     self.go_board.a0pos.board_mask = [[1 for y in range(19)]
        #                   for x in range(19)]
        #     self.go_board.a0pos.analyze_and_decompose(self.go_board.local_mask, self.go_board.agent)
        #     print(self.go_board.a0pos.segmentation)
        #     biggest_segment = None
        #     biggest_segment_size = 0
        #     for i in range(1, np.max(self.go_board.a0pos.segmentation)):
        #         segment = self.go_board.a0pos.segmentation == i
        #         segment_size = np.sum(segment)
        #         if biggest_segment_size < segment_size and segment_size < 20:
        #             biggest_segment_size = segment_size
        #             biggest_segment = segment
        #     self.go_board.local_mask = biggest_segment
        #     self.go_board.update()
        #     self.go_board.default_event_handler = self.go_board.add_mask

    def undo_move(self):
        self.go_board.undo_move()
        self.go_board.update()

    def clear_board(self):
        self.go_board.clear_board()
        self.go_board.update()

    def calculate_score(self):
        # self.go_board.position_tree.build_tree()  # current_node.calculate_score_and_ownership()
        # score = self.go_board.position_tree.current_node.calculated_score
        # ownership = self.go_board.position_tree.current_node.calculated_ownership
        self.thread = self.go_board.position_tree
        self.thread.update_signal.connect(self.update_tree)
        self.thread.error_signal.connect(self.update_tree)
        self.thread.start()

    def update_tree(self, error_message=None):
        # print("Updated")
        self.go_board.built_tree = True
        self.go_board.update()
        self.drawn_tree.create_tree(self.go_board.position_tree)
        if error_message is not None:
            print(error_message)
        self.update_buttons()
        # print("Done!!!")

    def select_directory(self):
        settings = QSettings("GoBoard", "Settings")
        last_directory = settings.value("last_directory")

        dialog = QFileDialog(self, "Select Directory")
        dialog.setFileMode(QFileDialog.Directory)
        # dialog.setOption(QFileDialog.ShowDirsOnly, True)
        # dialog.setFileMode(QFileDialog.AnyFile)

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
            elif self.selected_sgf.endswith(".sgf"):
                self.positions = [self.selected_sgf]
                # with open(self.selected_sgf, 'r') as f:
                #     self.positions = [f.read()]
                self.visualize_position()
            elif os.path.isdir(self.selected_sgf):
                self.positions = []
                for file in os.listdir(self.selected_sgf):
                    if file.endswith(".sgf"):
                        # with open(os.path.join(self.selected_sgf, file), 'r') as f:
                        #     self.positions.append(f.read())
                        self.positions.append(os.path.join(self.selected_sgf, file))
                self.visualize_position()

            settings.setValue("last_directory", selected_directory)
            # self.update_pinned_directories(selected_directory)

    def update_buttons(self):
        if not self.go_board.built_tree:
            self.label.setText("Game tree not built yet")
            return
        cgt_game = self.go_board.position_tree.current_node.cgt_game
        cgt_game_with_line_breaks = "<br>".join([str(cgt_game)[i:i + 40] for i in range(0, len(str(cgt_game)), 25)])
        # print("Tree built!", cgt_game, self.go_board.position_tree.current_node == self.go_board.position_tree.root)
        try:
            # split the cgt_game string into lines after every 25th character
            if not cgt_game.is_ok:
                self.label.setText(f'Game is not OK<br>{cgt_game_with_line_breaks}')
            elif not cgt_game.found_mt3:
                self.label.setText(f"<span style='color: red;'>Game is OK but still BAD</span><br>{cgt_game_with_line_breaks}")
            else:
                text = f'Temperature: {float(cgt_game.temp):.2f} points' if self.show_as_temp else f'Move value: {2 * float(cgt_game.temp) - 2:.2f} points'
                if self.go_board.gt is not None and self.go_board.position_tree.current_node == self.go_board.position_tree.root:
                    if almost_equal(float(2 * cgt_game.temp - 2), self.go_board.gt):
                        text += f"<br><span style='color: green;'>CORRECT!</span>"
                    else:
                        text += f"<br><span style='color: red;'>WRONG!</span> Ground truth: {self.go_board.gt}"
                text = f'{text}<br>Local score: {float(cgt_game.mean):.2f}'
                self.label.setText(text)
        except:
            self.label.setText(f"<span style='color: red;'>CGT calculations failed for:</span><br>{cgt_game_with_line_breaks}")
        evaluation = self.go_board.position_tree.eval_for_node()
        black_move_prob = evaluation.black_prob * (1 - evaluation.no_move_prob)
        white_move_prob = evaluation.white_prob * (1 - evaluation.no_move_prob)
        no_move_prob = evaluation.no_move_prob

        self.black_prob_label.setText(f"Black move prob: {round(100 * black_move_prob)}%")
        self.white_prob_label.setText(f"White move prob: {round(100 * white_move_prob)}%")
        self.no_move_prob_label.setText(f"No move prob: {round(100 * no_move_prob)}%")

    def visualize_position(self):
        current_position = self.positions[self.current_position_index]
        # if not self.go_board.from_pkl:
        #     current_position = json.loads(current_position)
        self.drawn_tree.clear_tree()
        self.go_board.built_tree = False
        self.go_board.visualize_position(current_position)
        self.update_buttons()

    def toggleFullScreen(self):
        if self.windowState() & Qt.WindowFullScreen:
            self.showNormal()
        else:
            self.showFullScreen()


if __name__ == "__main__":
    app = QApplication([])

    def excepthook(exc_type, exc_value, exc_tb):
        tb = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        print("error caught!:")
        print("error message:\n", tb)
        app.quit()
        # or QtWidgets.QApplication.exit(0)

    sys.excepthook = excepthook
    # with logging_context_manager():
    window = MainWindow()
    window.show()
    app.exec_()
