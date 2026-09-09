import os
import pickle
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import uvicorn
from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from src.agent import (
    LangChainAgentManager,
    Parser,
    Plain_text_chunker,
    RetrievalManager,
    School_rules_chunker,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 啟動階段...
    app.state.agent_manager = LangChainAgentManager()
    app.state.agent_manager.warnup()

    app.state.retrieval_manager = RetrievalManager()

    app.state.parser = Parser()

    app.state.plain_chunker = Plain_text_chunker()
    app.state.pdf_chunker = School_rules_chunker()

    yield

    # 關閉階段...
    temp_dir = Path("tmp")
    if temp_dir.exists():
        try:
            shutil.rmtree(temp_dir)
            print("[Shutdown] Successfully cleared temporary upload directory.")
        except Exception as e:
            print(f"[Warning][Shutdown] Failed to clear temp directory: {e}")

    try:
        if hasattr(app.state.retrieval_manager, "dense") and hasattr(
            app.state.retrieval_manager.dense, "client"
        ):
            if hasattr(app.state.retrieval_manager.dense.client, "close"):
                app.state.retrieval_manager.dense.client.close()
                print("[Shutdown] ChromaDB client closed successfully.")
    except Exception as e:
        print(f"[Warning][Shutdown] Error closing ChromaDB client: {e}")
    print("[Shutdown] Closed.")


app = FastAPI(title="LangChain Agent Web UI", lifespan=lifespan)


@app.get("/")
async def get_web_ui():
    return FileResponse("pages/index.html")


@app.get("/editor")
async def get_web_ui():
    return FileResponse("pages/editor.html")


@app.post("/api/store")
async def store(request: Request, files: List[UploadFile] = File(...)):
    # 從 FastAPI 的 application state 取得啟動時建立的處理元件。
    retrieval_manager = request.app.state.retrieval_manager
    plain_chunker = request.app.state.plain_chunker
    pdf_chunker = request.app.state.pdf_chunker
    parser = request.app.state.parser

    # 所有上傳檔案先寫入暫存目錄，方便既有 parser 以檔案路徑讀取。
    temp_dir = Path("tmp")
    temp_dir.mkdir(exist_ok=True)

    # 純文字檔直接讀取；PDF 則交給 parser 先轉成可切分的文字內容。
    wrapped_files = []
    needed2parse = []
    success_count = 0

    result_message = ""

    class CompatibleFile:
        def __init__(self, path):
            self.name = str(path)

    try:
        # 先保存所有上傳檔案，並依副檔名分派到不同處理流程。
        for file in files:
            file_ext = Path(file.filename).suffix.lower().lstrip(".")
            temp_file_path = temp_dir / file.filename
            with open(temp_file_path, "wb") as f:
                f.write(await file.read())
            if file_ext == "pdf":
                needed2parse.append(CompatibleFile(temp_file_path))
            else:  # txt or markdown
                wrapped_files.append(CompatibleFile(temp_file_path))

        # 處理純文字與 Markdown：讀取文字後切成 chunks，再寫入向量資料庫。
        for file in wrapped_files:
            filename = Path(file.name).name
            file_path = file.name

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    file_text = f.read()

                plain_metadata, plain_chunks = plain_chunker.chunk(
                    file_text, filename=filename
                )
                chunks = [chunk["content"] for chunk in plain_chunks]
                metadata = [
                    {"source": plain_metadata["title"], "article": chunk["article"]}
                    for chunk in plain_chunks
                ]
                if chunks:
                    msg = retrieval_manager.store(chunks=chunks, metadata=metadata)
                    if "Success" in msg:
                        success_count += 1
            except Exception as e:
                print(f"Failed to read {filename}: {e}")

        if needed2parse:
            # PDF 會在這裡直接轉成 Markdown、處理圖片，並由 parser 印出結果。
            print(f"\n===== Start Parsing {len(needed2parse)} PDF Files =====\n")
            parsed_raw_contents = parser.parse(files=needed2parse, contexts={})

            if isinstance(parsed_raw_contents, str):
                print(f"Error occurred while parsing PDF files: {parsed_raw_contents}")
                return {"status": "error", "message": parsed_raw_contents}

            for file_obj, markdown in zip(needed2parse, parsed_raw_contents):
                pdf_filename = Path(file_obj.name).name
                print(f"\n===== Markdown Output: {pdf_filename} =====\n")
                print(markdown)
                print(f"\n===== End Markdown Output: {pdf_filename} =====\n")

            for file_obj, content in zip(needed2parse, parsed_raw_contents):
                pdf_filename = Path(file_obj.name).name
                print(f"===== Processing PDF: {pdf_filename} =====")
                pdf_metadata, pdf_chunks = pdf_chunker.chunk(content)

                chunks = [chunk["content"] for chunk in pdf_chunks]
                metadata = [
                    {"source": pdf_metadata["title"], "article": chunk["article"]}
                    for chunk in pdf_chunks
                ]

                if chunks:
                    msg = retrieval_manager.store(chunks=chunks, metadata=metadata)
                    if "Success" in msg:
                        success_count += 1

        total_files = len(wrapped_files) + len(needed2parse)

    except Exception as e:
        # 上傳或批次處理發生未預期錯誤時，回傳 API 錯誤訊息。
        return {"status": "error", "message": str(e)}

    finally:
        # 不論成功或失敗，都刪除本次請求建立的暫存檔。
        for f_list in [wrapped_files, needed2parse]:
            for f in f_list:
                try:
                    if os.path.exists(f.name):
                        os.remove(f.name)
                except Exception as ce:
                    print(f"[Warning] Failed to remove temp file {f.name}: {ce}")

    return {
        "status": "success" if success_count > 0 else "failed",
        "message": f"Successfully processed {success_count}/{total_files} files.",
    }


@app.post("/api/clear")
async def clear_database(request: Request):
    retrieval_manager = request.app.state.retrieval_manager
    try:
        result_message = retrieval_manager.clear()
        if "Success" in result_message:
            return {"status": "success", "message": result_message}
        else:
            return {"status": "failed", "message": result_message}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/chat")
async def chat_stream(
    request: Request,
    question: str = Query(..., description="使用者輸入的問題"),
    thread_id: str = Query("default-thread", description="交談的 Thread ID"),
):
    agent_manager = request.app.state.agent_manager
    retrieval_manager = request.app.state.retrieval_manager

    USER_PROMPT = """
    Relevant Content:
    {context}

    User Question:
    {question}
    """.strip()

    try:
        relevant_chunks, sources = retrieval_manager.retrieve(message=question, topK=3)
        context = "\n\n".join(relevant_chunks)
    except Exception as e:
        context = f"Failed to Retrieve: {e}"

    async def combined_generator():
        if context.strip():
            yield f"[SEARCH_RESULTS]\n{context}\n|||CONTEXT_SPLITTER|||"
        else:
            yield "[SEARCH_RESULTS]\nNo relevant chunks retrieved.\n|||CONTEXT_SPLITTER|||"

        async for chunk in agent_manager.get_stream_response(
            USER_PROMPT.format(context=context, question=question), thread_id
        ):
            yield chunk

    return StreamingResponse(combined_generator(), media_type="text/plain")


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
