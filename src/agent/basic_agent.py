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
from datetime import datetime


class LangChainAgentManager:

    def _save_vlm_output(self, content: str ,img_path: str = ""):
        log_path = "tmp/vlm_output.txt"

        os.makedirs(
            os.path.dirname(log_path),
            exist_ok=True,
        )

        processing = getattr(
            self,
            "_last_image_processing_info",
            {},
        )

        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\n")
            f.write("=" * 80 + "\n")

            f.write(f"時間：{datetime.now()}\n")

            f.write(f"圖片：{img_path}\n")

            if processing:
                f.write(
                    f"處理：{processing['processing']}\n"
                )

                f.write(
                    f"原始尺寸："
                    f"{processing['original_size'][0]}x"
                    f"{processing['original_size'][1]}\n"
                )

                f.write(
                    f"處理後尺寸："
                    f"{processing['processed_size'][0]}x"
                    f"{processing['processed_size'][1]}\n"
                )

                f.write(
                    f"JPEG大小："
                    f"{processing['jpeg_size'] / 1024:.1f} KB\n"
                )

                f.write(
                    f"Base64大小："
                    f"{processing['base64_size'] / 1024:.1f} KB\n"
                )

            f.write("-" * 80 + "\n")
            f.write("VLM輸出：\n")
            f.write(content)
            f.write("\n")
            f.write("=" * 80 + "\n")

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
        max_size: int = 1600,
        quality: int = 85,
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
        original_size = image.size
        processing_steps = []

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
            processing_steps.append("透明背景轉白底")

        else:
            # 其他模式直接轉 RGB
            image = image.convert("RGB")
            processing_steps.append("RGB轉換")

        # 限制圖片最大尺寸
        image.thumbnail(
            (max_size, max_size),
            Image.Resampling.LANCZOS,
        )

        processing_steps.append(
            f"最大尺寸{max_size}x{max_size}"
        )
        processing_steps.append("LANCZOS縮放")

        # JPEG 壓縮
        buffer = io.BytesIO()

        image.save(
            buffer,
            format="JPEG",
            quality=quality,
            optimize=True,
        )

        processing_steps.append(
            f"JPEG(Quality={quality}, Optimize=True)"
        )

        image_bytes = buffer.getvalue()

        # Base64
        img_data = base64.b64encode(image_bytes).decode("utf-8")
        processing_steps.append("Base64編碼")

        processing_info = " → ".join(processing_steps)
        '''
        print("========== 圖片的處理資訊 ==========")

        print(
            f"[VLM Image] "
            f"原始尺寸={original_size}, "
            f"處理後尺寸={image.size}, "
            f"JPEG={len(image_bytes) / 1024:.1f} KB, "
            f"Base64={len(img_data) / 1024:.1f} KB"
        )

        print("========== END 圖片的處理資訊 ==========")

        self._last_image_processing_info = {
            "original_size": original_size,
            "processed_size": image.size,
            "jpeg_size": len(image_bytes),
            "base64_size": len(img_data),
            "processing": processing_info,
        }
        '''
        
        return f"data:image/jpeg;base64,{img_data}"

        #return img_data


    def get_img_caption(
        self,
        question: str,
        img_path_or_url: str,
    #    thread_id: str = "default-thread"
    ):
    #    config = {"configurable": {"thread_id": thread_id}}

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

        print("[VLM] Sending request to Ollama...")


        result = self.llm.invoke(
            [
                HumanMessage(
                    content=message_content
                )
            ]
        )

        '''
        result = self.llm.invoke(
            {"messages": [HumanMessage(content=message_content)]}, config
        )

        
        if isinstance(result, dict) and "messages" in result:
            messages = result["messages"]
            if messages:
                last_msg = messages[-1]
                if isinstance(last_msg, BaseMessage):
                    return last_msg.content
                return str(last_msg)
        '''
        print("[VLM] Response received.")

        if isinstance(result, BaseMessage):
            #print("========== VLM OUTPUT result.content ==========")
            #print(img_path_or_url) #檔案路徑
            #print(result.content)
            self._save_vlm_output(result.content, img_path_or_url)
            #print("========== END VLM OUTPUT ==========")
            return result.content
        #print("========== VLM OUTPUT result ==========")
        #print(result)
        self._save_vlm_output(str(result), img_path_or_url)
        #print("========== END VLM OUTPUT ==========")

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
