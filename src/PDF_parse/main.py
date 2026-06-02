import os
import pathlib
import pickle
import re
import shutil
from io import BytesIO
from threading import Thread

import jieba
import pymupdf4llm
import pymupdf.layout
import torch
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# NOTE 加上 layout 可以處理 header 和 footer，但是處理 table 會把合併儲存格的內容拆開來，導致表格亂掉
# NOTE 所以就視情況決定要不要加上 layout，之後可以實驗看看哪個比較好
# NOTE 或是直接另外寫把 header/footer 處理掉的功能
# NOTE 不過若要使用 ocr 功能，就必須要加上 layout 才行


DOC = "./tmp/中文測試文件.pdf"
IMAGES_OUTPUT_DIR = "./tmp/imgs/"
MD_OUTPUT_PATH = "./tmp/output.md"

BASE_PATH = "./tmp/datas"  # 存模型快取等資料
LLM_CACHE_PATH = os.path.join(BASE_PATH, "llm_cache")


def _make_sure_path_exists(p):
    if not os.path.exists(p):
        os.makedirs(p)


def setup():
    for p in [BASE_PATH, LLM_CACHE_PATH]:
        _make_sure_path_exists(p)


# 自動偵測是否支援 bfloat16
compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

# 設定量化參數，減少記憶體使用 / Set quantization settings for smaller memory usage
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 使用 4-bit 權重 / Use 4-bit weights
    bnb_4bit_use_double_quant=True,  # 啟用雙重量化 / Enable double quantization
    bnb_4bit_quant_type="nf4",  # 使用 nf4 量化類型 / Quantization type
    bnb_4bit_compute_dtype=compute_dtype,  # 使用 bfloat16 或 float16 進行計算 / Use bfloat16 or float16 for compute
)

# 載入微調好的語言模型（Gemma）/ Load pretrained LLM
llm = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path="unsloth/gemma-3-4b-it",
    quantization_config=bnb_config,
    dtype=compute_dtype,
    device_map="auto",
    low_cpu_mem_usage=True,
    cache_dir=LLM_CACHE_PATH,
)

processor = AutoProcessor.from_pretrained(
    "unsloth/gemma-3-4b-it",
    trust_remote_code=True,
    use_fast=True,
    cache_dir=LLM_CACHE_PATH,
)


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


def get_description_for_image(image_path):
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")

    chats = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": "請分析並簡短描述圖片內容，回覆內容僅須描述即可，不須包含任何問候或詢問句。",
                },
            ],
        },
    ]
    text_prompt = processor.apply_chat_template(chats, add_generation_prompt=True)
    inputs = processor(text=text_prompt, images=image, return_tensors="pt").to("cuda")

    generated_ids = llm.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,
    )
    generated_ids = generated_ids[:, inputs["input_ids"].shape[1] :]
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return generated_text


if __name__ == "__main__":
    md_text = pdf_to_markdown(DOC, IMAGES_OUTPUT_DIR)

    if not os.path.exists(MD_OUTPUT_PATH):
        open(MD_OUTPUT_PATH, "w").close()
    pathlib.Path(MD_OUTPUT_PATH).write_bytes(md_text.encode())

    # add captions for images
    caption = "This is a caption text for the image."  # caption 放 LLMs 對圖片的描述，像 LlamaIndex 那樣
    pattern = r"!\[.*?\]\((.*?)\)"

    def process_caption(match):
        image_path = match.group(
            1
        )  # NOTE 不用每張圖片都呼叫 LLMs，可以實作一個簡易分類器，只有在需要的圖片才呼叫 LLMs 來取得描述 (TODO：classifier)
        # NOTE 或是在進入點加入參數來決定是否要呼叫 LLMs 來取得描述，要特別告知使用者！
        caption = get_description_for_image(image_path)
        return f"![Image]({image_path})\n*({caption})*\n\n"

    with open(MD_OUTPUT_PATH, "r+", encoding="utf-8") as f:
        content = f.read()
        content = re.sub(pattern, process_caption, content)
        f.seek(0)
        f.write(content)
        f.truncate()
