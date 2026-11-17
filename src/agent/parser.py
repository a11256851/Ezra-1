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
from typing import List, Dict, Any, Optional
from PIL import Image

import jieba
import pymupdf4llm
import pymupdf
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

    #def _pdf2markdown(self, doc: str, img_output_dir: str) -> str :


    def _extract_tables(
        self,
        pdf_path: str,
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        使用 PyMuPDF 的 find_tables() 找出 PDF 中的表格，
        並直接轉成 Markdown。
        """

        result: Dict[int, List[Dict[str, Any]]] = {}

        doc = pymupdf.open(pdf_path)

        try:
            for page_number, page in enumerate(doc, start=1):

                page_tables = []

                try:
                    tabs = page.find_tables()

                    for table_index, table in enumerate(
                        tabs.tables,
                        start=1,
                    ):

                        bbox = table.bbox

                        # 優先使用 PyMuPDF Table.to_markdown()
                        if not hasattr(table, "to_markdown"):
                            raise RuntimeError(
                                f"頁面 {page_number} 表格 {table_index} 擷取失敗："
                                #f"PyMuPDF 版本不支援 to_markdown()"
                            )
                            print(
                                f"頁面 {page_number} 表格 {table_index} 擷取失敗："
                                #f"PyMuPDF 版本不支援 to_markdown()"
                            )
                        

                        markdown = table.to_markdown()
                            
                            

                        if not markdown:
                            continue

                        page_tables.append(
                            {
                                "page": page_number,
                                "index": table_index,
                                "bbox": (
                                    float(bbox[0]),
                                    float(bbox[1]),
                                    float(bbox[2]),
                                    float(bbox[3]),
                                ),
                                "markdown": markdown.strip(),
                            }
                        )

                        print(
                            f"頁面 {page_number} "
                            f"找到表格 {table_index}: "
                            f"bbox={bbox}"
                        )

                except Exception as e:
                    print(
                        f"頁面 {page_number} 表格擷取失敗: {e}"
                    )

                result[page_number] = page_tables

        finally:
            doc.close()

        return result
    
   

    @staticmethod
    def _is_bbox_inside_any_table(
        bbox,
        table_bboxes,
        threshold: float = 0.3,
    ) -> bool:
        """
        判斷文字 block 是否位於某個表格區域內。

        如果文字 bbox 有至少 threshold 比例
        落在表格 bbox 裡，就視為表格文字。
        """

        if not table_bboxes:
            return False

        x0, y0, x1, y1 = bbox

        block_area = max(
            0,
            (x1 - x0) * (y1 - y0),
        )

        if block_area <= 0:
            return False

        for table_bbox in table_bboxes:

            tx0, ty0, tx1, ty1 = table_bbox

            overlap_x = max(
                0,
                min(x1, tx1) - max(x0, tx0),
            )

            overlap_y = max(
                0,
                min(y1, ty1) - max(y0, ty0),
            )

            overlap_area = (
                overlap_x * overlap_y
            )

            overlap_ratio = (
                overlap_area / block_area
            )

            if overlap_ratio >= threshold:
                return True

        return False

    def _build_page_markdown(
        self,
        page_elements: List[Dict[str, Any]],
        page_images: List[Dict[str, Any]],
    ) -> str:
        """
        將文字、表格、embedded images
        按照 PDF 頁面座標交錯輸出。
        """

        elements = list(page_elements)

        # ==================================================
        # 加入 embedded images
        # ==================================================

        for image in page_images:

            bbox = image.get("bbox")

            if not bbox:
                continue

            elements.append(
                {
                    "type": "image",
                    "bbox": bbox,
                    "content": image,
                }
            )

        # ==================================================
        # 按照座標排序
        # ==================================================

        elements.sort(
            key=lambda element: (
                element["bbox"][1],
                element["bbox"][0],
            )
        )

        output_parts = []

        # ==================================================
        # 輸出
        # ==================================================

        for element in elements:

            element_type = element.get("type")

            # --------------------------------------------------
            # 普通文字
            # --------------------------------------------------

            if element_type == "text":

                text = element.get(
                    "content",
                    "",
                ).strip()

                if text:
                    output_parts.append(text)

            # --------------------------------------------------
            # 表格
            # --------------------------------------------------

            elif element_type == "table":

                table = element.get(
                    "content",
                    "",
                ).strip()

                if table:
                    output_parts.append(table)

            # --------------------------------------------------
            # 圖片
            # --------------------------------------------------

            elif element_type == "image":

                image = element.get(
                    "content"
                )

                if not image:
                    continue

                image_markdown = (
                    self._render_image_markdown(
                        image
                    )
                )

                if image_markdown.strip():
                    output_parts.append(
                        image_markdown.strip()
                    )

        return "\n\n".join(
            part
            for part in output_parts
            if part.strip()
        ).strip()

    def _pdf_to_page_chunks(
        self,
        pdf_path: str,
    ) -> List[Dict[str, Any]]:

        chunks: List[Dict[str, Any]] = []

        doc = pymupdf.open(pdf_path)

        try:
            for page_number, page in enumerate(doc, start=1):

                text_dict = page.get_text("dict")

                blocks = []

                for block in text_dict.get("blocks", []):

                    # 只處理文字 block
                    if block.get("type") != 0:
                        continue

                    bbox = block.get("bbox")

                    if not bbox:
                        continue

                    lines = []

                    for line in block.get("lines", []):

                        line_text = ""

                        for span in line.get("spans", []):
                            line_text += span.get("text", "")

                        if line_text:
                            lines.append(line_text)

                    text = "\n".join(lines).strip()

                    if not text:
                        continue

                    blocks.append(
                        {
                            "type": "text",
                            "bbox": (
                                float(bbox[0]),
                                float(bbox[1]),
                                float(bbox[2]),
                                float(bbox[3]),
                            ),
                            "text": text,
                        }
                    )

                chunks.append(
                    {
                        "page": page_number,
                        "text": page.get_text("text").strip(),
                        "blocks": blocks,
                        "page_boxes": [],
                    }
                )

        finally:
            doc.close()

        return chunks

    def _build_page_content(
        self,
        page_chunk: Dict[str, Any],
        page_tables: List[Dict[str, Any]],
        page_images: List[Dict[str, Any]],
    ) -> str:
        """
        將文字、表格、embedded images
        按照 PDF 頁面座標交錯輸出。

        注意：
        表格區域內的文字 block 不再另外輸出，
        避免表格內容重複。
        """

        elements = []

        # ==================================================
        # 取得所有表格 bbox
        # ==================================================

        table_bboxes = [
            table["bbox"]
            for table in page_tables
            if table.get("bbox")
        ]

        # ==================================================
        # 加入普通文字
        # ==================================================

        for block in page_chunk.get("blocks", []):

            bbox = block.get("bbox")

            if not bbox:
                continue

            # ----------------------------------------------
            # 如果文字位於表格內，就不要再當普通文字輸出
            # ----------------------------------------------

            if self._is_bbox_inside_any_table(
                bbox,
                table_bboxes,
            ):
                continue

            text = block.get("text", "").strip()

            if not text:
                continue

            elements.append(
                {
                    "type": "text",
                    "bbox": bbox,
                    "content": text,
                }
            )

        # ==================================================
        # 加入表格
        # ==================================================

        for table in page_tables:

            bbox = table.get("bbox")

            markdown = table.get(
                "markdown",
                "",
            ).strip()

            if not bbox or not markdown:
                continue

            elements.append(
                {
                    "type": "table",
                    "bbox": bbox,
                    "content": markdown,
                }
            )

        # ==================================================
        # 加入 embedded images
        # ==================================================

        for image in page_images:

            bbox = image.get("bbox")

            if not bbox:
                continue

            elements.append(
                {
                    "type": "image",
                    "bbox": bbox,
                    "content": image,
                }
            )

        # ==================================================
        # 按照頁面座標排序
        # ==================================================

        elements.sort(
            key=lambda element: (
                element["bbox"][1],
                element["bbox"][0],
            )
        )

        # ==================================================
        # 組合 Markdown
        # ==================================================

        output_parts = []

        for element in elements:

            element_type = element.get("type")

            # ----------------------------------------------
            # 文字
            # ----------------------------------------------

            if element_type == "text":

                text = element.get(
                    "content",
                    "",
                ).strip()

                if text:
                    output_parts.append(text)

            # ----------------------------------------------
            # 表格
            # ----------------------------------------------

            elif element_type == "table":

                table_markdown = element.get(
                    "content",
                    "",
                ).strip()

                if table_markdown:
                    output_parts.append(
                        table_markdown
                    )

            # ----------------------------------------------
            # 圖片
            # ----------------------------------------------

            elif element_type == "image":

                image = element.get("content")

                if not image:
                    continue

                image_markdown = (
                    self._render_image_markdown(
                        image
                    )
                )

                if image_markdown.strip():
                    output_parts.append(
                        image_markdown.strip()
                    )

        return "\n\n".join(
            output_parts
        ).strip()
    
    def _extract_embedded_images(
        self,
        pdf_path: str,
        output_dir: Path,
        ) -> Dict[int, List[Dict[str, Any]]]:
        
        #pdf_path = Path(pdf_path)
        #output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        #extracted_images = []

        doc = pymupdf.open(pdf_path)

        result: Dict[int, List[Dict[str, Any]]] = {}

        try:
            for page_index, page in enumerate(doc, start=1):
                page_images = []

                #print(f"\n--- 第 {page_index} 頁 embedded images ---")

                image_infos = page.get_image_info(xrefs=True)

                #print(f"\n--- 第 {page_num} 頁圖片檢查 ---")
                #print(f"找到 {len(images_infos)} 張 embedded image")
                #print( f"\n--- 第 {page_index} 頁 embedded images ---" ) 
                #print( f"圖片數量：{len(image_infos)}" )

                seen_xrefs = set()

                for image_index, image_info in enumerate(image_infos, start=1):
                    #xref = image_info[0]

                    xref = image_info.get( "xref", 0, ) 
                    bbox = image_info.get( "bbox" ) 
                    width = image_info.get( "width", 0, ) 
                    height = image_info.get( "height", 0, )

                    if not bbox: 
                        print( f"圖片 {image_index}: " f"沒有 bbox，跳過" ) 
                        continue

                    if not xref: 
                        print( f"圖片 {image_index}: " f"xref 不存在，跳過" ) 
                        continue

                    if xref in seen_xrefs:
                        print(
                            f"圖片 {image_index}："
                            f"xref={xref} 已擷取，跳過重複圖片"
                        )
                        continue

                    seen_xrefs.add(xref)

                    try:
                        image_data = doc.extract_image(xref)

                        image_ext = image_data["ext"]
                        image_bytes = image_data["image"]
                        real_width = image_data["width"]
                        real_height = image_data["height"]

                        output_path = (
                            output_dir
                            / (
                                f"{Path(pdf_path).stem}" 
                                f"-page{page_index}" 
                                f"-img{image_index}" 
                                f".{image_ext}" 
                            )
                        )

                        with open(
                            output_path,
                            "wb",
                        ) as f:
                            f.write(image_bytes)

                        image_record = { 
                            "page": page_index, 
                            "index": image_index, 
                            "xref": xref, 
                            "path": str( output_path ), 
                            "bbox": ( 
                                float(bbox[0]), 
                                float(bbox[1]), 
                                float(bbox[2]), 
                                float(bbox[3]), 
                            ), 
                                "width": real_width, 
                                "height": real_height, 
                                "ext": image_ext, 
                        }

                        '''
                        with open(output_path, "wb") as f:
                            f.write(image_bytes)
                        '''
                        page_images.append(image_record)

                        print(
                            f"圖片 {image_index}: " 
                            f"xref={xref}, " 
                            f"PDF尺寸=" 
                            f"{real_width}x{real_height}, " 
                            f"顯示位置="
                            f"{bbox}, "
                            f"格式={image_ext}, "
                            f"輸出={output_path}"
                        )


                        
                    except Exception as e:

                        print( f"圖片 {image_index} " 
                              f"擷取失敗：{e}" )

                result[page_index] = page_images

        finally:
            doc.close()

        return result

    @staticmethod 
    def _bbox_overlap( 
        bbox1, bbox2, 
        ) -> float:
            x0_1, y0_1, x1_1, y1_1 = bbox1 
            x0_2, y0_2, x1_2, y1_2 = bbox2 
            overlap_x = max( 
                0, 
                min(x1_1, x1_2) - max(x0_1, x0_2), 
                ) 
            overlap_y = max( 
                0, min(y1_1, y1_2) - max(y0_1, y0_2), 
                ) 
            intersection = ( overlap_x * overlap_y ) 
            area1 = max( 0, (x1_1 - x0_1) * (y1_1 - y0_1), ) 
            if area1 <= 0:
                return 0.0 
            return intersection / area1 
    # ======================================================== 
    # # 找到 page_box 對應的 embedded image 
    # # ======================================================== 
    def _find_image_for_box( self, 
                            box: Dict[str, Any], 
                            page_images: List[Dict[str, Any]], 
    ) -> Optional[Dict[str, Any]]: 
        box_bbox = box.get( "bbox" ) 
        if not box_bbox: 
            return None 
        best_image = None 
        best_overlap = 0.0 
        for image in page_images: 
            overlap = self._bbox_overlap( 
                box_bbox, 
                image["bbox"], 
                ) 
            if overlap > best_overlap: 
                    best_overlap = overlap 
                    best_image = image 
            # 至少有 50% 重疊才認定是同一張圖片 
            if best_overlap >= 0.5: 
                return best_image 
        return None

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
        vlm_img_path =( 
            self._resize_image_for_vlm(
                img_path,
                max_size=1600,
                quality=85,
            )
        )

        response = self.agent.get_img_caption(
            question = f"""
            圖片上下文： {context} 請忠實還原圖片內容，盡可能完整保留文字、數字、符號、物件、表格、圖表、流程、箭頭、位置與結構關係。 文字請盡可能準確辨識並保留，不要摘要或改寫。 僅描述圖片中可見資訊，不得猜測或補充；無法辨識處標示「[無法辨識]」。 讓未看到原圖的人也能透過描述理解圖片內容。 只輸出圖片描述。
            """.strip(),
            img_path_or_url=vlm_img_path,
        #    thread_id=f"vlm-{id(img_path)}",
        )
        return response

    def _analyze_embedded_images(
        self,
        page_images: Dict[int, List[Dict[str, Any]]],
        contexts: Dict[str, str],
        thread_id: str,
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        對所有擷取出的 embedded images 進行 VLM 描述。

        會直接把 caption 寫回每張圖片的 dictionary。
        """

        total_images = sum(
            len(images)
            for images in page_images.values()
        )

        print(f"\n開始分析 embedded images，共 {total_images} 張")

        analyzed_count = 0

        for page_number, images in page_images.items():
            for image in images:
                image_path = image.get("path")

                if not image_path:
                    image["caption"] = "[圖片路徑不存在]"
                    continue

                print(
                    f"\n分析圖片：第 {page_number} 頁，"
                    f"第 {image.get('index')} 張"
                )
                print(f"圖片路徑：{image_path}")

                try:
                    context = contexts.get(image_path, "無")

                    caption = self._get_description_for_img(
                        img_path=image_path,
                        context=context,
                        thread_id=thread_id,
                    )

                    if caption is None:
                        caption = ""

                    caption = str(caption).strip()

                    image["caption"] = caption

                    print(f"圖片描述：{caption[:300]}")

                except Exception as e:
                    print(f"VLM 圖片描述失敗：{image_path}")
                    traceback.print_exc()

                    image["caption"] = "[圖片描述失敗]"

                analyzed_count += 1

        print(
            f"\nembedded images 分析完成："
            f"{analyzed_count}/{total_images}"
        )

        return page_images

    def _render_image_markdown(
        self,
        image: Dict[str, Any],
    ) -> str:
        """
        將單張 embedded image 轉成 Markdown。
        """

        image_path = image.get("path", "")
        caption = image.get("caption", "").strip()

        if not image_path:
            return ""

        # 注意：
        # 這裡暫時使用實際檔案路徑。
        # 如果要給前端顯示，之後應改成 API URL 或相對路徑。
        markdown = (
            f"\n\n"
            f"![Embedded Image]({image_path})"
            f"\n\n"
        )

        if caption:
            markdown += (
                f"**圖片內容描述：**\n\n"
                f"{caption}"
                f"\n\n"
            )

        return markdown

    def _append_page_images_to_markdown(
        self,
        page_markdown: str,
        page_images: List[Dict[str, Any]],
    ) -> str:
        """
        將該頁的 embedded images 加入頁面 Markdown。

        目前因為 page_boxes 不可用，
        先將文字與圖片分開輸出，
        圖片依照頁面上的 y 座標排序。
        """

        if not page_images:
            return page_markdown.strip()

        sorted_images = sorted(
            page_images,
            key=lambda image: (
                image.get("bbox", (0, 0, 0, 0))[1],
                image.get("bbox", (0, 0, 0, 0))[0],
            ),
        )

        output_parts = []

        if page_markdown and page_markdown.strip():
            output_parts.append(page_markdown.strip())

        for image in sorted_images:
            image_markdown = self._render_image_markdown(image)

            if image_markdown.strip():
                output_parts.append(image_markdown.strip())

        return "\n\n".join(output_parts).strip()
    

    @staticmethod 
    def _get_box_text( 
            page_text: str, 
            box: Dict[str, Any], 
    ) -> str: 
        pos = box.get( "pos" ) 

        if not pos: 
            return "" 

        if len(pos) != 2: 
            return "" 

        start, stop = pos 

        if start is None or stop is None: 
            return "" 

        if start < 0 or stop < 0: 
            return "" 

        if start >= len(page_text): 
            return "" 

        stop = min( stop, len(page_text), ) 

        return page_text[ start:stop ]
   
    def parse(self, files, contexts, thread_id: str = "default-thread") -> List[str]:
        if not files:
            return "Please upload file first"

        if contexts is None:
            contexts = {}

        raw_contents = []

        for file in files:
            filename = Path(file.name).name
            file_path = file.name

            try:
                if ( not os.path.isfile(file_path) or os.path.getsize(file_path) == 0 ):
                    raise ValueError("PDF file is missing or empty")
                
                print("\n"
                      +"=" * 70)
                print(f"開始解析 PDF: {filename}")

                print("=" * 70)

                page_chunks = ( self._pdf_to_page_chunks( file_path ) )

                page_tables = self._extract_tables(file_path)

                total_tables = sum(len(tables)for tables in page_tables.values())

                print( f"PDF 頁數： {len(page_chunks)}" )

                page_images = ( self._extract_embedded_images( file_path, img_output_dir, ) )

                total_images = sum( len(images) for images in page_images.values() )

                print(f"\n實際擷取 tables："f"{total_tables} 張")

                print( f"\n實際擷取 embedded images：" f"{total_images} 張" )

                page_images = self._analyze_embedded_images(
                    page_images=page_images,
                    contexts=contexts,
                    thread_id=thread_id,
                )

                document_parts = []

                for page_number, page_chunk in enumerate(
                    page_chunks,
                    start=1,
                ):

                    current_page_tables = page_tables.get(
                        page_number,
                        []
                    )

                    current_page_images = page_images.get(
                        page_number,
                        []
                    )

                    page_markdown = self._build_page_content(
                        page_chunk=page_chunk,
                        page_tables=current_page_tables,
                        page_images=current_page_images,
                    )

                    if page_markdown.strip():

                        document_parts.append(
                            f"<!-- Page {page_number} -->\n\n"
                            f"{page_markdown.strip()}"
                        )

                # ==================================================
                # 所有頁面最後只組一次
                # ==================================================

                content = "\n\n".join(
                        document_parts
                )
                '''
                content = "\n\n".join(
                                    part
                                    for part in document_parts
                                    if part.strip()
                                )
                '''
                print("\n" + "=" * 70)
                """
                print(f"===== Final Markdown: {filename} =====")
                print(content)
                print("===== End Final Markdown =====")
                """
                print("=" * 70 + "\n")

                raw_contents.append(content)
                
                #print(f"\n===== Markdown: {filename} =====\n{content}\n===== End Markdown: {filename} =====\n")
                #raw_contents.append(content)
            except Exception as e:
                traceback.print_exc()

                return (
                    f"Failed to read"  
                    f"{filename}: {e}"
                )

                #return f"Failed to read {filename}: {e}"

        return raw_contents
