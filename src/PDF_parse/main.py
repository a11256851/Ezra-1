import os
import pathlib
import re

import pymupdf4llm
import pymupdf.layout

# NOTE 加上 layout 可以處理 header 和 footer，但是處理 table 會把合併儲存格的內容拆開來，導致表格亂掉
# NOTE 所以就視情況決定要不要加上 layout，之後可以實驗看看哪個比較好
# NOTE 或是直接另外寫把 header/footer 處理掉的功能
# NOTE 不過若要使用 ocr 功能，就必須要加上 layout 才行

DOC = "./test/中文測試文件.pdf"
IMAGES_OUTPUT_DIR = "./test/imgs/"
MD_OUTPUT_PATH = "./test/output.md"


def pdf_to_markdown(doc: str, image_output_dir: str) -> str:
    md_text = pymupdf4llm.to_markdown(
        doc=doc,
        write_images=True,
        image_path=image_output_dir,
        image_format="png",
        dpi=300,
        # header=False,  # do not include header in output
        # footer=False,  # do not include footer in output
        # NOTE using OCR should install https://github.com/tesseract-ocr/tesseract first
        # use_ocr=False,
        # use_ocr=True,
        # ocr_language="eng" # +chi_tra for traditional Chinese
    )
    return md_text


if __name__ == "__main__":
    md_text = pdf_to_markdown(DOC, IMAGES_OUTPUT_DIR)

    if not os.path.exists(MD_OUTPUT_PATH):
        open(MD_OUTPUT_PATH, "w").close()
    pathlib.Path(MD_OUTPUT_PATH).write_bytes(md_text.encode())

    # add captions for images
    caption = "This is a caption text for the image."  # 之後 caption 可以放 LLMs 對圖片的描述，像 LlamaIndex 那樣
    pattern = r"!\[.*?\]\((assets/images/[^)]+)\)"

    def process_caption(match):
        image_path = match.group(
            1
        )  # NOTE 不用每張圖片都呼叫 LLMs，可以實作一個簡易分類器，只有在需要的圖片才呼叫 LLMs 來取得描述 (TODO：classifier)
        # NOTE 或是在進入點加入參數來決定是否要呼叫 LLMs 來取得描述，要特別告知使用者！
        # caption = get_description_for_image(image_path) # 這裡可以呼叫 LLMs 來取得圖片描述 (TODO：call LLMs for description)
        return f"![Image]({image_path})\n*({caption})*\n\n"

    with open(MD_OUTPUT_PATH, "r+", encoding="utf-8") as f:
        content = f.read()
        content = re.sub(pattern, process_caption, content)
        f.seek(0)
        f.write(content)
        f.truncate()
