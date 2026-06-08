import os
import pickle
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import uvicorn
from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from src.agent import LangChainAgentManager, RetrievalManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 啟動階段...
    app.state.agent_manager = LangChainAgentManager()
    app.state.agent_manager.warnup()

    app.state.retrieval_manager = RetrievalManager()

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
    return FileResponse("index.html")


@app.post("/api/store")
async def store(request: Request, files: List[UploadFile] = File(...)):
    retrieval_manager = request.app.state.retrieval_manager
    temp_dir = Path("tmp")
    temp_dir.mkdir(exist_ok=True)

    wrapped_files = []

    class CompatibleFile:
        def __init__(self, path):
            self.name = str(path)

    try:
        for file in files:
            temp_file_path = temp_dir / file.filename
            with open(temp_file_path, "wb") as f:
                f.write(await file.read())
            wrapped_files.append(CompatibleFile(temp_file_path))

        result_message = retrieval_manager.store(wrapped_files)

    except Exception as e:
        return {"status": "error", "message": str(e)}

    finally:
        for wf in wrapped_files:
            try:
                if os.path.exists(wf.name):
                    os.remove(wf.name)
            except Exception as ce:
                print(f"[Warning] Failed to remove temp file {wf.name}: {ce}")

    return {
        "status": "success" if "Success." in result_message else "failed",
        "message": result_message,
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
        context = retrieval_manager.retrieve(message=question, topK=3)
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
