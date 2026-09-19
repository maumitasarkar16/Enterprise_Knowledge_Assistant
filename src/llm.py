"""OpenAI helpers for routing, general chat, query rewriting and grounded generation."""

import logging
from openai import OpenAI

logger = logging.getLogger(__name__)

NOT_FOUND_MESSAGE = (
    "I couldn't find this information in the provided enterprise documents."
)

class OpenAILLM:
    """Thin wrapper around OpenAI Responses API.

    The application has two modes:
    - GENERAL: normal conversational/general-knowledge answer.
    - ENTERPRISE: retrieval-augmented answer strictly grounded in local documents.
    """

    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is missing. Copy .env.example to .env and add your key."
            )
        self.client = OpenAI(api_key=api_key)
        self.model = model

    @staticmethod
    def _history_text(history: list[dict], max_messages: int) -> str:
        recent = history[-max_messages:]
        return "\n".join(
            f"{m.get('role', 'user').upper()}: {m.get('content', '')}"
            for m in recent
        )

    def route_query(self, question: str, history: list[dict], max_messages: int) -> str:
        """Classify a request as GENERAL or ENTERPRISE.

        ENTERPRISE includes company-specific/internal questions even when the answer
        may not exist in the corpus. Follow-ups inherit enterprise context.
        """
        # Cheap deterministic handling for obvious social chat. This avoids routing
        # greetings through RAG and improves UX.
        normalized = " ".join(question.lower().strip().split())
        greeting_prefixes = (
            "hi", "hello", "hey", "good morning", "good afternoon",
            "good evening", "thanks", "thank you"
        )
        if not history and normalized.startswith(greeting_prefixes):
            return "GENERAL"

        transcript = self._history_text(history, max_messages)
        prompt = f"""
Classify the LATEST USER MESSAGE into exactly one label:

ENTERPRISE
- asks about this organisation/company, its employees, internal policies, travel,
  leave, expenses, security, production access, support/SLA, backups, products,
  internal procedures, or information expected to come from enterprise documents;
- asks a follow-up to an earlier ENTERPRISE/company-policy discussion;
- asks company-specific information even if it may be absent from the documents
  (for example: "Who is our CEO?" or "What is our maternity leave policy?").

GENERAL
- greetings, casual conversation, personal introductions;
- public/general knowledge unrelated to the organisation;
- general explanations such as Python, RAG, geography, mathematics, etc.;
- follow-ups to an earlier GENERAL discussion.

When uncertain whether a question is about the organisation's own rules/data,
prefer ENTERPRISE.

Return ONLY: GENERAL or ENTERPRISE.

RECENT CONVERSATION:
{transcript or "(none)"}

LATEST USER MESSAGE:
{question}
""".strip()

        response = self.client.responses.create(
            model=self.model, input=prompt, store=False
        )
        label = (response.output_text or "").strip().upper()
        return "GENERAL" if label == "GENERAL" else "ENTERPRISE"

    def answer_general(self, question: str, history: list[dict], max_messages: int) -> str:
        """Answer normal conversation/general knowledge without enterprise RAG."""
        transcript = self._history_text(history, max_messages)
        instructions = """
You are a helpful general assistant inside an enterprise knowledge application.
Answer normal greetings, casual conversation and public/general-knowledge questions
naturally and concisely. Do not pretend that general knowledge comes from the
enterprise documents. If the user asks about the organisation's own policies or
internal facts, do not invent them; say that such questions should be answered
from the enterprise knowledge base.
""".strip()
        user_input = (
            f"RECENT CONVERSATION:\n{transcript or '(none)'}\n\n"
            f"LATEST USER MESSAGE:\n{question}"
        )
        response = self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=user_input,
            store=False,
        )
        return (response.output_text or "").strip() or "How can I help?"

    def rewrite_query(self, question, history, max_messages):
        """Resolve follow-up references into a standalone enterprise retrieval query."""
        if not history:
            return question.strip()

        transcript = self._history_text(history, max_messages)
        prompt = f"""
Rewrite the latest question as a concise standalone ENTERPRISE search query.
Resolve pronouns/references using the conversation.
Do not answer. Do not add facts.
Return only the rewritten query.

CONVERSATION:
{transcript}

LATEST QUESTION:
{question}
""".strip()
        try:
            response = self.client.responses.create(
                model=self.model, input=prompt, store=False
            )
            rewritten = (response.output_text or "").strip()
            return rewritten or question.strip()
        except Exception:
            logger.exception("Query rewrite failed; falling back to original query")
            return question.strip()

    def answer_enterprise(self, question, context):
        """Generate only from retrieved enterprise context."""
        system = f"""
You are an enterprise knowledge assistant.

RULES:
1. Answer ONLY from CONTEXT.
2. Do not use outside knowledge or assumptions for enterprise facts.
3. Every factual enterprise claim must use one or more source markers like [S1].
4. If context is insufficient, reply exactly:
   {NOT_FOUND_MESSAGE}
5. If sources conflict, say so and cite both.
6. Preserve exact dates, limits, policy conditions and names.
""".strip()

        response = self.client.responses.create(
            model=self.model,
            instructions=system,
            input=f"QUESTION:\n{question}\n\nCONTEXT:\n{context}",
            store=False,
        )
        return (response.output_text or "").strip() or NOT_FOUND_MESSAGE
