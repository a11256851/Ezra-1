import os
import pathlib
import pickle
import re
import shutil
import traceback
import fitz
from io import BytesIO
from pathlib import Path
from threading import Thread
from typing import Dict, List, Tuple
from PIL import Image

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

img_output_dir.mkdir(parents=True, exist_ok=True)


class Parser:
    def __init__(self):
        self.agent = LangChainAgentManager()
        img_output_dir.mkdir(parents=True, exist_ok=True)

    def _pdf2markdown(self, doc: str, img_output_dir: str) -> str:

        """
        # 先直接檢查 PDF 原始文字
        pdf = fitz.open(doc)

        print("\n========== PDF 原始內容檢查 ==========")
        print(f"PDF 頁數：{len(pdf)}")

        for page_num, page in enumerate(pdf, start=1):
            text = page.get_text("text")
            images = page.get_images(full=True)

            print(f"\n--- 第 {page_num} 頁 ---")
            print(f"文字長度：{len(text)}")
            print(f"圖片數量：{len(images)}")
            print(f"文字預覽：{repr(text[:300])}")

        pdf.close()
        """


    
        md_text = pymupdf4llm.to_markdown(
            doc=doc,
            write_images=False,  # 不直接寫入圖片檔案，改用 extract_embedded_images 擷取真正的 embedded images
            image_path=img_output_dir,
            image_format="png",
            dpi=150,
            # header=False,  # do not include header in output
            # footer=False,  # do not include footer in output
            # NOTE using OCR should install https://github.com/tesseract-ocr/tesseract first
            # use_ocr=False,
            use_ocr=True,
            #ocr_language="eng" # +chi_tra for traditional Chinese
        )

        """
        print("\n========== Markdown 檢查 ==========")
        print(f"Markdown 長度：{len(md_text)}")
        print(repr(md_text[:1000]))
        print("========== END ==========\n")
        """

        return md_text

    
    def _extract_embedded_images(
        self,
        pdf_path: str,
        output_dir: str,
        ) -> List[str]:
        
        """
        從 PDF 中擷取真正的 embedded images。
        不將 PDF 頁面渲染成整頁圖片。
        """

        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        extracted_images = []

        doc = fitz.open(pdf_path)

        try:
            for page_num, page in enumerate(doc, start=1):
                images = page.get_images(full=True)

                print(f"\n--- 第 {page_num} 頁圖片檢查 ---")
                print(f"找到 {len(images)} 張 embedded image")

                for image_num, image_info in enumerate(images, start=1):
                    xref = image_info[0]

                    try:
                        image_data = doc.extract_image(xref)

                        image_ext = image_data["ext"]
                        image_bytes = image_data["image"]
                        width = image_data["width"]
                        height = image_data["height"]

                        output_path = (
                            output_dir
                            / f"{pdf_path.stem}-page{page_num}-img{image_num}.{image_ext}"
                        )

                        with open(output_path, "wb") as f:
                            f.write(image_bytes)

                        print(
                            f"圖片 {image_num}: "
                            f"xref={xref}, "
                            f"尺寸={width}x{height}, "
                            f"格式={image_ext}, "
                            f"路徑={output_path}"
                        )

                        extracted_images.append(str(output_path))

                    except Exception as e:
                        print(
                            f"擷取第 {page_num} 頁第 {image_num} 張圖片失敗: {e}"
                        )

        finally:
            doc.close()

        return extracted_images
    
    def _resize_image_for_vlm(
        self,
        img_path: str,
        max_size: int = 1600,
        quality: int = 85,
        ) -> str:
        img_path = Path(img_path)

        with Image.open(img_path) as img:
            img = img.convert("RGB")

            # 保持比例縮小
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

            output_path = img_path.with_suffix(".jpg")

            img.save(
                output_path,
                format="JPEG",
                quality=quality,
                optimize=True,
            )
        return str(output_path)


    def _get_description_for_img(
        self, img_path, context: str = "無", thread_id: str = "default-thread"
    ) -> str:
        vlm_img_path = self._resize_image_for_vlm(
        img_path,
        max_size=1600,
        quality=85,
        )

        response = self.agent.get_img_caption(
            question = f"""
            圖片上下文： {context} 請忠實還原圖片內容，盡可能完整保留文字、數字、符號、物件、表格、圖表、流程、箭頭、位置與結構關係。 文字請盡可能準確辨識並保留，不要摘要或改寫。 僅描述圖片中可見資訊，不得猜測或補充；無法辨識處標示「[無法辨識]」。 讓未看到原圖的人也能透過描述理解圖片內容。 只輸出圖片描述。
            """.strip(),
            img_path_or_url=vlm_img_path,
        #    thread_id=f"vlm-{id(img_path)}",
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
                """
                md_text = self._pdf2markdown(file_path, img_output_dir)
                
                caption = "This is a caption text for the image."
                pattern = r"!\[.*?\]\((.*?)\)"

                def _process_caption(statement):
                    img_path = statement.group(1)
                    caption = self._get_description_for_img(
                        img_path,
                        context=contexts.get(img_path, "無"),
                    #    thread_id=thread_id,
                    )
                    return f"![Image]({img_path})\n*({caption})*\n\n"

                content = re.sub(pattern, _process_caption, md_text)
                """

                md_text=self._pdf2markdown(
                    file_path,
                    img_output_dir,
                )

                print(
                    f"\n========== Markdown 檢查 =========="
                )
                print(f"Markdown 長度：{len(md_text)}")
                print(repr(md_text[:1000]))
                print(
                    f"========== END Markdown ==========\n"
                )

                extracted_images = self._extract_embedded_images(
                file_path,
                img_output_dir,
                )

                print(
                    f"\n========== Embedded Images =========="
                )
                print(
                    f"實際擷取圖片數量：{len(extracted_images)}"
                )

                for img_path in extracted_images:
                    print(f"圖片：{img_path}")

                print(
                    f"========== END Embedded Images ==========\n"
                )

                content = md_text

                for img_path in extracted_images:

                    caption = self._get_description_for_img(
                        img_path,
                        context=contexts.get(img_path, "無"),
                        thread_id=thread_id,
                    )

                    content += (
                        f"\n\n"
                        f"![Image]({img_path})\n"
                        f"*({caption})*\n"
                    )

                print(
                    f"\n===== Final Markdown: {filename} ====="
                )
                print(content)
                print(
                    f"===== End Final Markdown: {filename} =====\n"
                )

                #print(f"\n===== Markdown: {filename} =====\n{content}\n===== End Markdown: {filename} =====\n")
                raw_contents.append(content)
            except Exception as e:
                traceback.print_exc()

                return (
                    f"Failed to read {filename}: {e}"
                )

                #return f"Failed to read {filename}: {e}"

        return raw_contents
