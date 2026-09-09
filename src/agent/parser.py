import os
import pathlib
import pickle
import re
import shutil
import traceback
from io import BytesIO
from pathlib import Path
from threading import Thread
from typing import Dict, List, Tuple

import jieba
import pymupdf4llm
#import pymupdf.layout
import torch
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from .basic_agent import LangChainAgentManager

# NOTE 加上 layout 可以處理 header 和 footer，但是處理 table 會把合併儲存格的內容拆開來，導致表格亂掉
# NOTE 所以就視情況決定要不要加上 layout，之後可以實驗看看哪個比較好
# NOTE 或是直接另外寫把 header/footer 處理掉的功能
# NOTE 不過若要使用 ocr 功能，就必須要加上 layout 才行


curent_dir = Path(__file__).parent
root_dir = curent_dir.parent.parent
tmp_dir = root_dir / "tmp"
img_output_dir = tmp_dir / "imgs"


class Parser:
    def __init__(self):
        self.agent = LangChainAgentManager()
        img_output_dir.mkdir(parents=True, exist_ok=True)

    def _pdf2markdown(self, doc: str, img_output_dir: str) -> str:
        md_text = pymupdf4llm.to_markdown(
            doc=doc,
            write_images=True,
            image_path=img_output_dir,
            image_format="png",
            dpi=150,
            # header=False,  # do not include header in output
            # footer=False,  # do not include footer in output
            # NOTE using OCR should install https://github.com/tesseract-ocr/tesseract first
            # use_ocr=False,
            # use_ocr=True,
            # ocr_language="eng" # +chi_tra for traditional Chinese
        )
        return md_text

    def _get_description_for_img(
        self, img_path, context: str = "無", thread_id: str = "default-thread"
    ) -> str:
        response = self.agent.get_img_caption(
            question=f"原圖片上下文：\n{context}\n請分析並簡短描述圖片內容，回覆內容僅須描述即可，不須包含任何問候或詢問句。".strip(),
            img_path_or_url=img_path,
            thread_id=thread_id,
        )
        return response

    def parse(self, files, contexts, thread_id: str = "default-thread") -> List[str]:
        if not files:
            return "Please upload file first"

        raw_contents = []

        for file in files:
            filename = Path(file.name).name
            file_path = file.name

            try:
                if not os.path.isfile(file_path) or os.path.getsize(file_path) == 0:
                    raise ValueError("PDF file is missing or empty")

                md_text = self._pdf2markdown(file_path, img_output_dir)
                caption = "This is a caption text for the image."
                pattern = r"!\[.*?\]\((.*?)\)"

                def _process_caption(statement):
                    img_path = statement.group(1)
                    caption = self._get_description_for_img(
                        img_path,
                        context=contexts.get(img_path, "無"),
                        thread_id=thread_id,
                    )
                    return f"![Image]({img_path})\n*({caption})*\n\n"

                content = re.sub(pattern, _process_caption, md_text)
                print(f"\n===== Markdown: {filename} =====\n{content}\n===== End Markdown: {filename} =====\n")
                raw_contents.append(content)
            except Exception as e:
                traceback.print_exc()
                return f"Failed to read {filename}: {e}"

        return raw_contents
