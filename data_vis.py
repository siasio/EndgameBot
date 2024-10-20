from pyx import color

from Sgfs_to_pdf.scripts.factory import DocumentLayout, NoneFactory as NF, ConstantFactory as CF, LoopedFactory as LF, \
    MetaHeaderFactory, DateFactory, RangeFactory
from Sgfs_to_pdf.scripts.marking import Marking
from Sgfs_to_pdf.scripts.positions_to_pdfs import graphic_grayed_position
from Sgfs_to_pdf.scripts.sgf_to_position import sgf_to_position
from Sgfs_to_pdf.scripts.sheet_creator import create_sh


def prepare_layout(num_sgfs_per_page):
    return DocumentLayout(NF(), NF(), CF(num_sgfs_per_page))


def visualize(sgfs, pdf_path, num_sgfs_per_page=15, board_color=color.rgb(224/255, 184/255, 135/255)):
    canvases = [graphic_grayed_position(
        sgf_to_position(
            sgf,
            swap=False,
            white_marking=Marking.square,
            smart_coords=True,
            solution=False,
            square=True,
            MIN_SKIP=1
        ),
        board_color=board_color
    ) for sgf in sgfs]
    layout = prepare_layout(num_sgfs_per_page)
    d = create_sh(canvases, layout, vertical=False, save_boxes=False)
    d.writePDFfile(pdf_path)
