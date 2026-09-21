"""Reassemble the two-sample qualitative figure used in the main paper.

Input: the final image- and video-domain qualitative PDFs in ``res/``; no model
weights are required. Output: the main-paper PDF and PNG composition.
"""
from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parent
ASSET_ROOT = ROOT.parent / "res"
OUT = ROOT / "outputs/main_qualitative"
IMAGE_PDF = ASSET_ROOT / "qualitative_p_AGM.pdf"
VIDEO_PDF = ASSET_ROOT / "qualitative_v_AGM.pdf"
OUTPUT_PDF = OUT / "3-2_AGM.pdf"
OUTPUT_PNG = OUT / "3-2_AGM.png"

INK = (37 / 255, 50 / 255, 57 / 255)
RM = (40 / 255, 126 / 255, 138 / 255)
AGM = (194 / 255, 119 / 255, 62 / 255)


def centered_text(page, rect, text, size, color, bold=True):
    result = page.insert_textbox(
        rect,
        text,
        fontsize=size,
        fontname="hebo" if bold else "helv",
        color=color,
        align=fitz.TEXT_ALIGN_CENTER,
        lineheight=0.92,
        overlay=True,
    )
    if result < 0:
        raise RuntimeError(f"Text does not fit: {text!r} in {rect}")


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    image_doc = fitz.open(IMAGE_PDF)
    video_doc = fitz.open(VIDEO_PDF)
    image_page = image_doc[0]
    video_page = video_doc[0]

    image_boxes = [fitz.Rect(item["bbox"]) for item in image_page.get_image_info(xrefs=True)]
    video_boxes = [fitz.Rect(item["bbox"]) for item in video_page.get_image_info(xrefs=True)]
    if len(image_boxes) != 36 or len(video_boxes) != 36:
        raise RuntimeError("Expected a 6-row by 6-column layout in each source PDF")

    # Keep all panels equally sized and place the two masks side by side.
    source_order = [0, 1, 2, 4, 3, 5]
    column_widths = [64.0] * 6
    column_gap = 5.5
    margin = 3.0
    columns = []
    x = margin
    for width in column_widths:
        columns.append(fitz.Rect(x, 0, x + width, 1))
        x += width + column_gap
    page_width = x - column_gap + margin
    max_tile_height = max(column_widths)
    header_top = 1.0
    header_height = 17.0
    first_label_top = 19.0
    label_height = 8.5
    first_tile_top = 28.0
    row_gap = 8.5
    second_label_top = first_tile_top + max_tile_height + row_gap
    second_tile_top = second_label_top + label_height
    page_height = second_tile_top + max_tile_height + 2.0

    output = fitz.open()
    page = output.new_page(width=page_width, height=page_height)

    mask_band = fitz.Rect(columns[4].x0 - 3.0, 0, columns[5].x1 + 3.0, page_height)
    page.draw_rect(mask_band, color=None, fill=(0.965, 0.972, 0.974), overlay=False)
    page.draw_line(
        fitz.Point(columns[4].x0 - 3.0, 0),
        fitz.Point(columns[4].x0 - 3.0, page_height),
        color=(0.82, 0.85, 0.86),
        width=0.55,
        overlay=False,
    )

    headers = [
        ("Input", INK),
        ("Ground\ntruth", INK),
        ("RM\nsegmentation", RM),
        ("AGM\nsegmentation", AGM),
        ("RM\nmask", RM),
        ("AGM\nmask", AGM),
    ]
    for box, (label, color) in zip(columns, headers):
        centered_text(
            page,
            fitz.Rect(box.x0, header_top, box.x1, header_top + header_height),
            label,
            size=7.4,
            color=color,
        )

    rows = [
        (image_doc, image_boxes[18:24], "COCO / 297", first_label_top, first_tile_top),
        (video_doc, video_boxes[12:18], "MOVi-E / 127", second_label_top, second_tile_top),
    ]
    for source_doc, source_boxes, row_label, label_top, tile_top in rows:
        page.insert_text(
            fitz.Point(columns[0].x0, label_top + 6.2),
            row_label,
            fontsize=7.1,
            fontname="hebo",
            color=INK,
            overlay=True,
        )

        ordered_source_boxes = [source_boxes[index] for index in source_order]
        for panel_index, (column, source_box) in enumerate(zip(columns, ordered_source_boxes)):
            tile_size = column_widths[panel_index]
            vertical_offset = (max_tile_height - tile_size) / 2.0
            destination = fitz.Rect(
                column.x0,
                tile_top + vertical_offset,
                column.x1,
                tile_top + vertical_offset + tile_size,
            )
            page.show_pdf_page(destination, source_doc, 0, clip=source_box, keep_proportion=True)
            if panel_index >= 4:
                page.draw_rect(
                    destination,
                    color=RM if panel_index == 4 else AGM,
                    width=1.0,
                    overlay=True,
                )

    metadata = output.metadata
    metadata.update({"title": "", "author": "", "subject": "", "keywords": "", "creator": ""})
    output.set_metadata(metadata)
    output.save(OUTPUT_PDF, garbage=4, deflate=True, clean=True)
    output.close()
    image_doc.close()
    video_doc.close()

    check = fitz.open(OUTPUT_PDF)
    pixmap = check[0].get_pixmap(matrix=fitz.Matrix(4, 4), alpha=False)
    pixmap.save(OUTPUT_PNG)
    print(f"pdf={OUTPUT_PDF}")
    print(f"png={OUTPUT_PNG}")
    print(f"page={check[0].rect.width:.2f}x{check[0].rect.height:.2f} pt")
    check.close()


if __name__ == "__main__":
    build()
