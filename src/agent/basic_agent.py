import argparse
import base64
import json
import os
import subprocess
import sys
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver


class LangChainAgentManager:
    def __init__(self):
        print("Initializing LangChainAgentManager...")
        self.llm = self._create_llm()
        self.checkpointer = InMemorySaver()
        self.agent = self._create_agent()

    def _create_llm(self) -> ChatOllama:
        base_url = os.getenv("OLLAMA_BASE_URL")
        model = os.getenv("OLLAMA_MODEL", "gemma4:31b-cloud")

        kwargs = {"model": model, "temperature": 0}
        if base_url:
            kwargs["base_url"] = base_url

        return ChatOllama(**kwargs)

    def _create_agent(self) -> Any:
        tools = []
        agent = create_agent(
            self.llm,
            tools,
            checkpointer=self.checkpointer,
            system_prompt=("你是 Langchain Agent ，請盡你所能回答使用者問題。"),
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

    def get_img_caption(
        self, question: str, img_path_or_url: str, thread_id: str = "default-thread"
    ):
        config = {"configurable": {"thread_id": thread_id}}
        if os.path.exists(img_path_or_url):
            with open(img_path_or_url, "rb") as img_file:
                encoded_string = base64.b64encode(img_file.read()).decode("utf-8")
            ext = os.path.splitext(img_path_or_url)[1].replace(".", "")
            mime_type = (
                f"image/{ext}"
                if ext in ["png", "jpg", "jpeg", "webp"]
                else "image/jpeg"
            )
            img_data = f"data:{mime_type};base64,{encoded_string}"
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
