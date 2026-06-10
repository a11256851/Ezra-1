import re
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter


class ChunkingStrategy(ABC):
    @abstractmethod
    def chunk(self, text: str) -> Tuple[Dict, List[Dict]]:
        pass


class Plain_text_chunker(ChunkingStrategy):
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50
        )

    def chunk(self, text, filename: str = "Unknown") -> Tuple[Dict, List[Dict]]:
        metadata = {"title": filename}
        chunks = []

        if text and text.strip():
            split_docs = self.text_splitter.split_text(text)
            vaild_docs = [doc for doc in split_docs if doc.strip()]

            for i, doc in enumerate(vaild_docs):
                chunks.append(
                    {
                        "article": f"{filename} _ Part {i+1}",
                        "content": doc,
                    }
                )
        else:
            print(f"[Warning] File {filename} is empty.")

        return metadata, chunks


class School_rules_chunker(ChunkingStrategy):
    def chunk(self, text) -> Tuple[Dict, List[Dict]]:
        metadata = {}
        chunks = []

        lines = text.split("\n")

        title_line = lines[0].replace("`", "")
        title_match = re.search(r"^##\s*(.*)", title_line)
        if title_match:
            metadata["title"] = title_match.group(1).strip()
        else:
            metadata["title"] = "Untitled"

        dates = []
        date_pattern = re.compile(r"^\s*([`\d一二三四五六七八九十百]+.*年.*日.*通過.*)")
        i = 1
        while i < len(lines):
            line = lines[i].strip()
            if not line:  # Skip empty lines
                i += 1
                continue

            # line may start with noise like " )" and other characters, so we need to reomve all non-date characters before matching
            line = re.sub(
                r"^[^\d一二三四五六七八九十百]*", "", line
            )  # Remove leading non-date characters

            # Clean up backticks for matching
            cleaned_line = line.replace("`", "")
            if date_pattern.match(cleaned_line):
                dates.append(cleaned_line)
            else:
                # If the line doesn't match the date pattern, stop processing dates.
                break
            i += 1
        metadata["date"] = dates[-1] if len(dates) >= 1 else "Unknown Date"
        if metadata["date"] != "Unknown Date":
            metadata["date"] = metadata["date"].replace(
                " ", ""
            )  # Remove spaces for easier regex matching
        metadata["year"] = (
            re.search(r"([`\d一二三四五六七八九十百]+)年", metadata["date"]).group(1)
            if re.search(r"([`\d一二三四五六七八九十百]+)年", metadata["date"])
            else "Unknown Year"
        )
        metadata["month"] = (
            re.search(r"([`\d一二三四五六七八九十]+)月", metadata["date"]).group(1)
            if re.search(r"([`\d一二三四五六七八九十]+)月", metadata["date"])
            else "Unknown Month"
        )
        metadata["day"] = (
            re.search(r"([`\d一二三四五六七八九十]+)日", metadata["date"]).group(1)
            if re.search(r"([`\d一二三四五六七八九十]+)日", metadata["date"])
            else "Unknown Day"
        )

        # Extract Chunks (Articles)
        current_chunk = None
        # Pattern for "第一條", "一、", etc.
        # 如果最高階層的樣式是 "第一條"、"第二條"、...，就用 patternA； 如果最高階層的樣式是 "一、"、"二、"、...，就用 patternB
        article_patternA = re.compile(
            r"^\s*(?:-\s*)?(第?[一二三四五六七八九十百]+條)\s*"
        )
        article_patternB = re.compile(r"^\s*(?:-\s*)?([一二三四五六七八九十百]+、)\s*")
        # Try pattern A first, if it doesn't match, try pattern B
        # Check the first few non-empty lines to determine the pattern
        patten_check = r"- 第[\d一二三四五六七八九十百]+條"
        article_pattern = (
            article_patternA if re.search(patten_check, text) else article_patternB
        )
        for line in lines[i:]:
            line = line.strip()
            if not line:
                continue

            # Clean up backticks for matching
            cleaned_line = line.replace("`", "")
            match = article_pattern.match(cleaned_line)

            if match:
                if current_chunk:
                    chunks.append(current_chunk)

                article_title = metadata["title"] + " > " + match.group(1)
                content_start = match.end()
                current_chunk = {
                    "article": article_title,
                    "content": cleaned_line[content_start:].strip(),
                }
            elif current_chunk:
                # Also clean backticks from content lines
                current_chunk["content"] += "\n" + line.replace("`", "")

        if current_chunk:
            chunks.append(current_chunk)

        return metadata, chunks
