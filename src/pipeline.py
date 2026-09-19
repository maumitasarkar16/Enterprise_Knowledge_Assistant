"""End-to-end dual-mode assistant orchestration.

GENERAL mode:
    User -> Router -> General LLM

ENTERPRISE mode:
    User -> Router -> Conversation Rewrite -> Dense + BM25 -> RRF ->
    Cross-Encoder -> Relevance Guard -> Grounded LLM + Sources
"""

from src.chunking import chunk_documents
from src.document_loader import load_documents
from src.embeddings import LocalEmbeddingModel
from src.llm import OpenAILLM, NOT_FOUND_MESSAGE
from src.models import AnswerResult
from src.retrieval import HybridRetriever
from src.reranker import CrossEncoderReranker
from src.vector_store import ChromaVectorStore

class RAGPipeline:
    def __init__(self, settings):
        self.settings = settings
        self.indexed_chunk_count = 0

    def initialize(self):
        documents = load_documents(self.settings.data_dir)
        chunks = chunk_documents(
            documents,
            self.settings.chunk_size_words,
            self.settings.chunk_overlap_words,
        )
        if not chunks:
            raise ValueError("No chunks were produced.")

        self.embedding_model = LocalEmbeddingModel(
            self.settings.embedding_model, self.settings.huggingface_api_key
        )
        self.vector_store = ChromaVectorStore(
            self.settings.chroma_dir, self.settings.collection_name
        )
        embeddings = self.embedding_model.embed_documents([c.text for c in chunks])
        self.vector_store.rebuild(chunks, embeddings)

        self.retriever = HybridRetriever(
            chunks, self.vector_store, self.embedding_model, self.settings.rrf_k
        )
        self.reranker = CrossEncoderReranker(
            self.settings.reranker_model, self.settings.huggingface_api_key
        )
        self.llm = OpenAILLM(
            self.settings.openai_api_key, self.settings.openai_chat_model
        )
        self.indexed_chunk_count = len(chunks)

    def _build_context(self, ranked):
        blocks, sources = [], []
        for i, item in enumerate(ranked, 1):
            citation = f"S{i}"
            source = item.chunk.metadata["source"]
            chunk_index = int(item.chunk.metadata["chunk_index"])
            blocks.append(
                f"[{citation}] Source: {source}, chunk: {chunk_index}\n"
                f"{item.chunk.text}"
            )
            sources.append({
                "citation_id": citation,
                "source": source,
                "chunk_index": chunk_index,
                "rerank_score": item.rerank_score,
                "preview": (
                    item.chunk.text[:350] + "..."
                    if len(item.chunk.text) > 350 else item.chunk.text
                ),
            })
        return "\n\n---\n\n".join(blocks), sources

    def answer(self, question, history):
        if not question or not question.strip():
            return AnswerResult("Please enter a question.", mode="GENERAL")

        mode = self.llm.route_query(
            question, history, self.settings.max_history_messages
        )

        if mode == "GENERAL":
            answer = self.llm.answer_general(
                question, history, self.settings.max_history_messages
            )
            return AnswerResult(answer=answer, mode="GENERAL")

        retrieval_query = self.llm.rewrite_query(
            question, history, self.settings.max_history_messages
        )
        candidates = self.retriever.search(
            retrieval_query,
            self.settings.dense_top_k,
            self.settings.sparse_top_k,
            self.settings.hybrid_top_k,
        )
        ranked = self.reranker.rerank(
            retrieval_query, candidates, self.settings.rerank_top_k
        )
        best = ranked[0].rerank_score if ranked else 0.0

        if not ranked or best < self.settings.min_rerank_score:
            return AnswerResult(
                NOT_FOUND_MESSAGE,
                mode="ENTERPRISE",
                retrieval_query=retrieval_query,
                best_rerank_score=best,
                sources=[],
            )

        grounded = [
            r for r in ranked
            if r.rerank_score >= self.settings.min_rerank_score
        ]
        context, sources = self._build_context(grounded)
        answer = self.llm.answer_enterprise(question, context)

        if answer.strip() == NOT_FOUND_MESSAGE:
            sources = []

        return AnswerResult(
            answer=answer,
            mode="ENTERPRISE",
            retrieval_query=retrieval_query,
            best_rerank_score=best,
            sources=sources,
        )
