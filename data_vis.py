from pyx import color, document

from Sgfs_to_pdf.scripts.factory import DocumentLayout, NoneFactory as NF, ConstantFactory as CF, LoopedFactory as LF, \
    MetaHeaderFactory, DateFactory, RangeFactory
from Sgfs_to_pdf.scripts.marking import Marking
from Sgfs_to_pdf.scripts.positions_to_pdfs import graphic_grayed_position
from Sgfs_to_pdf.scripts.sgf_to_position import sgf_to_position
from Sgfs_to_pdf.scripts.sheet_creator import create_sh
from color import Color


def prepare_layout(num_sgfs_per_page):
    return DocumentLayout(NF(), NF(), CF(num_sgfs_per_page))


def visualize(sgfs, pdf_path, num_sgfs_per_page=15, board_color=Color(224/255, 184/255, 135/255)):
    rgb = board_color
    board_gray = board_color * 0.9 #color.rgb(0.9 * board_color.r, 0.9 * board_color.g, 0.9 * board_color.b)
    canvases = [graphic_grayed_position(
        sgf_to_position(
            sgf,
            swap=False,
            white_marking=Marking.square,
            smart_coords=True,
            solution=True,
            square=True,
            MIN_SKIP=1,
            rotate=False,
        ),
        board_color=board_color,
        board_gray=board_gray,
    ) for sgf in sgfs]
    layout = prepare_layout(num_sgfs_per_page)
    d = create_sh(canvases, layout, vertical=False, save_boxes=False)
    d.writePDFfile(pdf_path)


def visualize_one(sgf, pdf_path, board_color=Color(224/255, 184/255, 135/255)):
    rgb = board_color
    board_gray = board_color * 0.9 #color.rgb(0.9 * board_color.r, 0.9 * board_color.g, 0.9 * board_color.b)
    canva = graphic_grayed_position(
        sgf_to_position(
            sgf,
            swap=False,
            white_marking=None,
            gray_marking=Marking.triangle,
            smart_coords=True,
            solution=True,
            square=True,
            MIN_SKIP=1,
            rotate=False,
        ),
        board_color=board_color,
        board_gray=board_gray,
    )
    p = document.page(canva, paperformat=None, rotated=False, centered=0,
                      margin=0, fittosize=1)
    small_d = document.document()
    small_d.append(p)
    small_d.writePDFfile(pdf_path)
