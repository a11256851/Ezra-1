import pickle
import shutil
import uuid
from pathlib import Path
from typing import List

import chromadb
import jieba
from chromadb import Documents, EmbeddingFunction, Embeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

curent_dir = Path(__file__).parent
EMBEDDING_MODEL_CACHE_PATH = curent_dir / "embedding_models"
CHROMA_PATH = curent_dir / "chromaDB"
SPARSE_PATH = curent_dir / "sparse"
BM25_PATH = SPARSE_PATH / "bm25_index.pkl"
CORPUS_PATH = SPARSE_PATH / "corpus_list.pkl"


class E5EmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name: str = "intfloat/multilingual-e5-large"):
        print(f"[Embedding] Loading model: {model_name}...")
        self.model = SentenceTransformer(
            model_name, cache_folder=EMBEDDING_MODEL_CACHE_PATH
        )

    def encode_queries(self, queries):
        return self.model.encode(
            [f"query: {q}" for q in queries], normalize_embeddings=True
        ).tolist()

    def encode_documents(self, docs):
        return self.model.encode(
            [f"passage: {d}" for d in docs], normalize_embeddings=True
        ).tolist()

    def __call__(self, input: Documents) -> Embeddings:
        return self.encode_documents(input)


class VectorDBManager:
    def __init__(self, collection_name: str = "4rag"):
        self.embedding_function = E5EmbeddingFunction()
        CHROMA_PATH.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )
        print(
            f"[VectorDB] Initialized. Already loaded {self.collection.count()} chunks."
        )

    def store(self, chunks, metadatas, ids):
        try:
            self.collection.add(documents=chunks, metadatas=metadatas, ids=ids)
        except Exception as e:
            print(f"[Error][VectorDB] Failed to update vector DB: {e}")
            return False
        return True

    def retrieve(self, query: str, topK: int = 5) -> List[str]:
        query_vec = self.embedding_function.encode_queries([query])
        results = self.collection.query(
            query_embeddings=query_vec,
            n_results=topK,
        )
        if results["documents"]:
            return results["documents"][0]
        return []


class SparseManager:
    def __init__(self):
        self._chunks = []
        self._bm25 = None
        SPARSE_PATH.mkdir(parents=True, exist_ok=True)

        if BM25_PATH.exists() and CORPUS_PATH.exists():
            try:
                with open(BM25_PATH, "rb") as f:
                    self._bm25 = pickle.load(f)
                with open(CORPUS_PATH, "rb") as f:
                    self._chunks = pickle.load(f)
                print(
                    f"[SparseDB] Loaded existing BM25 index with {len(self._chunks)} chunks."
                )
            except Exception as e:
                print(f"[Warning][SparseDB] Failed to load existing index: {e}.")

    @property
    def chunks(self):
        return self._chunks

    def store(self, chunks):
        try:
            self._chunks = self._chunks + chunks
            print(f"[SparseDB] Tokenizing {len(self._chunks)} chunks using jieba...")
            tokenized_chunks = [list(jieba.cut(chunk)) for chunk in self._chunks]
            self._bm25 = BM25Okapi(tokenized_chunks)

            with open(BM25_PATH, "wb") as f:
                pickle.dump(self._bm25, f)
            with open(CORPUS_PATH, "wb") as f:
                pickle.dump(self._chunks, f)
            print(f"[SparseDB] BM25 index successfully updated and saved.")
        except Exception as e:
            print(f"[Error][SparseDB] Failed to update BM25: {e}")
            return False
        return True

    def retrieve(self, query: str, topK: int = 5) -> List[str]:
        if not self._bm25:
            return []
        tokenized_query = list(jieba.cut(query))
        scores = self._bm25.get_scores(tokenized_query)
        rank = sorted(zip(self._chunks, scores), key=lambda x: x[1], reverse=True)[
            :topK
        ]
        return [chunk for chunk, score in rank if score > 0]


class RetrievalManager:
    def __init__(self):
        self.dense = VectorDBManager()
        self.sparse = SparseManager()

    def store(self, chunks, metadata):
        if not chunks or len(chunks) == 0:
            return "Error: can't extract text from all files."

        ids = [str(uuid.uuid4()) for _ in chunks]
        if not self.dense.store(chunks, metadata, ids):
            return "Failed to update vector DB."
        if not self.sparse.store(chunks):
            return "Failed to update BM25, but successfully update vector DB."

        return "Success."

    @staticmethod
    def _reciprocal_rank_fusion(*ranked_lists, k: int = 60) -> List[str]:
        scores = {}
        for rl in ranked_lists:
            for rank, doc_content in enumerate(rl, start=1):
                if doc_content not in scores:
                    scores[doc_content] = 0
                scores[doc_content] += 1.0 / (k + rank)
        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in fused]

    def retrieve(self, message: str, topK: int = 5) -> str:
        relevant_chunks_sparse = self.sparse.retrieve(query=message, topK=topK)
        relevant_chunks_dense = self.dense.retrieve(query=message, topK=topK)

        relevant_chunks = "\n\n".join(
            self._reciprocal_rank_fusion(relevant_chunks_sparse, relevant_chunks_dense)[
                :topK
            ]
        )
        return relevant_chunks

    def clear(self) -> str:
        errors = []

        try:
            collection_name = self.dense.collection.name
            self.dense.client.delete_collection(name=collection_name)
            self.dense.collection = self.dense.client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.dense.embedding_function,
                metadata={"hnsw:space": "cosine"},
            )
            print("[VectorDB] Collection cleared and re-created successfully.")
        except Exception as e:
            errors.append(f"VectorDB clear error: {e}")

        try:
            self.sparse._chunks = []
            self.sparse._bm25 = None

            if BM25_PATH.exists():
                BM25_PATH.unlink()
            if CORPUS_PATH.exists():
                CORPUS_PATH.unlink()

            print("[SparseDB] BM25 index and corpus files deleted successfully.")
        except Exception as e:
            errors.append(f"SparseDB clear error: {e}")

        if errors:
            return f"Clear completed with errors: {'; '.join(errors)}"
        return "Success. All databases have been completely cleared."
