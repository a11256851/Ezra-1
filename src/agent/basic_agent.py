import argparse
import base64
import json
import os
import subprocess
import sys
import io
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from io import BytesIO
from PIL import Image


class LangChainAgentManager:
    def __init__(
        self, system_prompt: str = "你是 Langchain Agent ，請盡你所能回答使用者問題。"
    ):
        print("Initializing LangChainAgentManager...")
        self.llm = self._create_llm()
        self.checkpointer = InMemorySaver()
        self.agent = self._create_agent(system_prompt)

    def _create_llm(self) -> ChatOllama:
        base_url = os.getenv("OLLAMA_BASE_URL")
        model = os.getenv("OLLAMA_MODEL", "gemma4:31b-cloud")

        kwargs = {"model": model, "temperature": 0}
        if base_url:
            kwargs["base_url"] = base_url

        return ChatOllama(**kwargs)

    def _create_agent(self, system_prompt) -> Any:
        tools = []
        agent = create_agent(
            self.llm,
            tools,
            checkpointer=self.checkpointer,
            system_prompt=(system_prompt),
        )
        return agent

    def get_response(self, question: str, thread_id: str = "default-thread") -> str:
        config = {"configurable": {"thread_id": thread_id}}
        result = self.agent.invoke(
            {"messages": [HumanMessage(content=question)]}, config
        )

        if isinstance(result, dict) and "messages" in result:
            messages = result["messages"]
            if messages:
                last_msg = messages[-1]
                if isinstance(last_msg, BaseMessage):
                    return last_msg.content
                return str(last_msg)

        return str(result)

    async def get_stream_response(
        self, question: str, thread_id: str = "default-thread"
    ):
        config = {"configurable": {"thread_id": thread_id}}

        for chunk, metadata in self.agent.stream(
            {"messages": [HumanMessage(content=question)]},
            config,
            stream_mode="messages",
        ):
            if hasattr(chunk, "content") and chunk.content:
                yield chunk.content
    """
    def _prepare_image_for_vlm(
        self,
        img_path: str,
        max_size: int = 1600,
        quality: int = 85,
    ) -> str:

        with Image.open(img_path) as img:
            # PNG 可能是 RGBA，JPEG 不支援 alpha
            if img.mode != "RGB":
                img = img.convert("RGB")

            # 保持比例縮小
            img.thumbnail(
                (max_size, max_size),
                Image.Resampling.LANCZOS,
            )

            buffer = BytesIO()

            img.save(
                buffer,
                format="JPEG",
                quality=quality,
                optimize=True,
            )

            encoded_string = base64.b64encode(
                buffer.getvalue()
            ).decode("utf-8")

        return f"data:image/jpeg;base64,{encoded_string}"
    """

    def _prepare_image_for_vlm(
        self,
        img_path_or_url: str,
        max_size: int = 1536,
        quality: int = 80,
    ) -> str:
        """
        Prepare an image for VLM:
        - load image
        - resize if necessary
        - convert to RGB
        - JPEG compress
        - return base64 string
        """

        image = Image.open(img_path_or_url)

        # 如果圖片有透明背景，先放到白色背景
        if image.mode in ("RGBA", "LA") or (
            image.mode == "P" and "transparency" in image.info
        ):
            image = image.convert("RGBA")

            background = Image.new(
                "RGB",
                image.size,
                "white",
            )

            background.paste(
                image,
                mask=image.getchannel("A"),
            )

            image = background

        else:
            # 其他模式直接轉 RGB
            image = image.convert("RGB")

            # 限制圖片最大尺寸
        image.thumbnail(
            (max_size, max_size),
            Image.Resampling.LANCZOS,
        )

        # JPEG 壓縮
        buffer = io.BytesIO()

        image.save(
            buffer,
            format="JPEG",
            quality=quality,
            optimize=True,
        )

        image_bytes = buffer.getvalue()

        # Base64
        img_data = base64.b64encode(image_bytes).decode("utf-8")

        print(
            f"[VLM Image] "
            f"size={image.size}, "
            f"JPEG={len(image_bytes) / 1024:.1f} KB, "
            f"Base64={len(img_data) / 1024:.1f} KB"
        )

        return img_data


    def get_img_caption(
        self, question: str, img_path_or_url: str, thread_id: str = "default-thread"
    ):
        config = {"configurable": {"thread_id": thread_id}}

        if os.path.exists(img_path_or_url):
#            with open(img_path_or_url, "rb") as img_file:
#                encoded_string = base64.b64encode(img_file.read()).decode("utf-8")
#            ext = os.path.splitext(img_path_or_url)[1].replace(".", "")
#            mime_type = (
#                f"image/{ext}"
#                if ext in ["png", "jpg", "jpeg", "webp"]
#                else "image/jpeg"
#            )
#            img_data = f"data:{mime_type};base64,{encoded_string}"
            img_data=self._prepare_image_for_vlm(img_path_or_url)
        else:
            img_data = img_path_or_url
        message_content = [
            {"type": "text", "text": question},
            {
                "type": "image_url",
                "image_url": {"url": img_data},
            },
        ]
        result = self.agent.invoke(
            {"messages": [HumanMessage(content=message_content)]}, config
        )

        if isinstance(result, dict) and "messages" in result:
            messages = result["messages"]
            if messages:
                last_msg = messages[-1]
                if isinstance(last_msg, BaseMessage):
                    return last_msg.content
                return str(last_msg)

        return str(result)

    def warnup(self):
        print("Warning up LLM model...")
        try:
            self.llm.invoke("ping")
            print("LLM model warned up successfully.")
        except Exception as e:
            print(f"Failed to warn up LLM: {e}")


def example():
    agent_manager = LangChainAgentManager()

    parser = argparse.ArgumentParser(
        description="LangChain ReAct agent with Ollama and document search (multi-turn with memory)"
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Question to ask the agent. If omitted, enters interactive mode.",
    )
    args = parser.parse_args()

    thread_id = os.getenv("THREAD_ID", "default-thread")

    if args.question:
        question = args.question
        print("Processing query...\n")

        try:
            reply = agent_manager.get_response(args.question, thread_id)
            print(reply)

        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc()
            sys.exit(1)
    else:
        print("LangChain Agent (Multi-turn with Memory)")
        print(f"Thread ID: {thread_id}")
        print("Type your questions or 'exit' to quit\n")

        try:
            while True:
                try:
                    question = input("You: ").strip()
                except EOFError:
                    print("\nGoodbye!")
                    break

                if not question:
                    continue

                if question.lower() in ["exit", "quit", "bye"]:
                    print("Goodbye!")
                    break

                print("Agent: ", end="", flush=True)

                try:
                    reply = agent_manager.get_response(question, thread_id)
                    print(reply)

                except Exception as e:
                    print(f"Error: {e}", file=sys.stderr)

                print()  # Add spacing between turns

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            sys.exit(0)




if __name__ == "__main__":
    example()
