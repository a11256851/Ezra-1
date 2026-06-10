import json
import os
import pickle
import shutil
from pathlib import Path
from typing import List

from langchain_core.messages import HumanMessage
from tqdm.autonotebook import tqdm

from experiments.exp import Experiment
from src.agent import (
    LangChainAgentManager,
    Parser,
    Plain_text_chunker,
    RetrievalManager,
    School_rules_chunker,
)


class Exp1(Experiment):
    def __init__(self):
        super().__init__()
        self.test_set = self.data_dir / "test1.json"
        self.output = self.results_dir / "exp1.json"

        self.retrieval_manager = RetrievalManager()  # 預設資料已載入

        self.SYSTEM_PROMPT = """
                請根據**問題**，從使用者提的**參考資料**找出與問題相關的內容，直接提供精簡並準確的解答，避免與問題無關的內容。
                請使用繁體中文回覆。
                如果資料未提供相關資訊，請回答「未提供相關資訊」。
                如果你不知道正確答案，則回答「不知道」
                """.strip()
        self.USER_PROMPT = """
                ### 問題:
                {question}

                ### 參考資料:
                {context}
                """.strip()
        self.JUDGE_SYSTEM_PROMPT = """
                請根據**問題**、**標準解答**、**標準來源**、**模型回答**和**模型檢索來源**，
                作為一名裁判對模型檢索和生成的能力進行評分，回饋格式是「檢索 1~10 分，生成 1~10 分」，盡可能簡短回覆。
                """.strip()
        self.JUDGE_PROMPT = """
                ### 問題
                {query}
                ### 標準解答
                {answer}
                ### 標準來源
                {source}
                ### 模型回答
                {model_answer}
                ### 模型檢索來源
                {model_source}
                """.strip()

        self.agent_manager = LangChainAgentManager(self.SYSTEM_PROMPT)
        self.judge_manager = LangChainAgentManager(self.JUDGE_SYSTEM_PROMPT)

    def showInfo(self):
        print("Exp1: 校規Q&A, questions: 70 (請確認校規資訊已載入)")

    def eval(self):
        try:
            evaluation_results = {}
            with open(self.test_set, "r") as f:
                qa_dataset = json.load(f)
                print(f"dataset load successful. total {len(qa_dataset)} Q&A.\n")
            for item_id, data in tqdm(qa_dataset.items(), desc="LLM Judge progress"):
                question = data.get("q", "no question")
                answer = data.get("a", "no answer")
                source = data.get("where", "no source")

                try:
                    relevant_chunks, src = self.retrieval_manager.retrieve(
                        message=question, topK=3
                    )
                    context = "\n\n".join(relevant_chunks)
                except Exception as e:
                    context = f"Failed to Retrieve: {e}"
                    src = "No relevant sources"

                res = self.agent_manager.get_response(
                    question=self.USER_PROMPT.format(question=question, context=context)
                )
                judge_res = self.judge_manager.get_response(
                    question=self.JUDGE_PROMPT.format(
                        query=question,
                        answer=answer,
                        source=source,
                        model_answer=res,
                        model_source=src,
                    )
                )

                evaluation_results[item_id] = {
                    "question": question,
                    "answer": answer,
                    "source": source,
                    "model_answer": res,
                    "model_source": src,
                    "eavl": judge_res,
                }
            with open(self.output, "w", encoding="utf-8") as f:
                json.dump(evaluation_results, f, ensure_ascii=False, indent=4)
        except FileNotFoundError:
            print(f"Dataset {test_set} not found\n")
        except json.JSONDecodeError:
            print("JSON format decode error\n")
