import os

import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, StreamingResponse

from src.agent import agent_manager

app = FastAPI(title="LangChain Agent Web UI")


@app.get("/")
async def get_web_ui():
    return FileResponse("index.html")


@app.get("/api/chat")
async def chat_stream(
    question: str = Query(..., description="使用者輸入的問題"),
    thread_id: str = Query("default-thread", description="交談的 Thread ID"),
):
    generator = agent_manager.get_stream_response(question, thread_id)

    return StreamingResponse(generator, media_type="text/plain")


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
