"""Streamlit UI for the dual-mode Enterprise Knowledge Assistant."""

import logging
import streamlit as st

from src.config import get_settings
from src.pipeline import RAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

st.set_page_config(
    page_title="Enterprise Knowledge Assistant",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 Enterprise Knowledge Assistant")
st.caption(
    "GENERAL assistant + ENTERPRISE Advanced RAG | "
    "Dense Retrieval + BM25 + RRF + Cross-Encoder + Memory + Citations"
)

@st.cache_resource(show_spinner="Loading models and indexing local documents...")
def load_pipeline():
    pipeline = RAGPipeline(get_settings())
    pipeline.initialize()
    return pipeline

try:
    assistant = load_pipeline()
except Exception as exc:
    st.error(f"Application initialization failed: {exc}")
    st.info("Please check .env, requirements, and data/sample_docs.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("System Status")
    st.success(f"Indexed chunks: {assistant.indexed_chunk_count}")
    st.write("Routing: `GENERAL / ENTERPRISE`")
    st.write(f"LLM: `{assistant.settings.openai_chat_model}`")
    st.write(f"Embedding: `{assistant.settings.embedding_model}`")
    st.write(f"Reranker: `{assistant.settings.reranker_model}`")
    st.divider()
    st.caption(
        "GENERAL handles greetings and public knowledge. "
        "ENTERPRISE uses only the local knowledge base."
    )
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("mode") and message["role"] == "assistant":
            st.caption(f"Mode: {message['mode']}")
        if message.get("sources"):
            with st.expander("Sources used"):
                for source in message["sources"]:
                    st.markdown(
                        f"**[{source['citation_id']}] {source['source']}** "
                        f"(chunk {source['chunk_index']}, "
                        f"relevance {source['rerank_score']:.3f})"
                    )
                    st.caption(source["preview"])

question = st.chat_input("Ask a general question or a question about enterprise documents...")

if question:
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Routing and answering..."):
            try:
                result = assistant.answer(question, history)
            except Exception as exc:
                logging.exception("Question answering failed")
                st.error(f"Unable to answer: {exc}")
                st.stop()

        st.markdown(result.answer)
        st.caption(f"Mode: {result.mode}")

        if result.sources:
            with st.expander("Sources used", expanded=True):
                for source in result.sources:
                    st.markdown(
                        f"**[{source['citation_id']}] {source['source']}** "
                        f"(chunk {source['chunk_index']}, "
                        f"relevance {source['rerank_score']:.3f})"
                    )
                    st.caption(source["preview"])

        with st.expander("Diagnostics"):
            st.write(f"Route: `{result.mode}`")
            if result.mode == "ENTERPRISE":
                st.write(f"Standalone retrieval query: `{result.retrieval_query}`")
                st.write(f"Best reranker score: `{result.best_rerank_score:.4f}`")
                st.write(
                    "Grounding decision: "
                    f"`{'ANSWERED' if result.sources else 'NOT_FOUND'}`"
                )
            else:
                st.write("RAG retrieval: `SKIPPED`")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result.answer,
        "mode": result.mode,
        "sources": result.sources,
    })
