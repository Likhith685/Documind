import json
import time
import urllib.error
import urllib.request
from typing import Any, Iterator, Optional

from langchain_community.vectorstores import FAISS

try:
    from langchain_ollama import ChatOllama
except ImportError:
    from langchain_community.chat_models import ChatOllama

DEFAULT_MODEL = "tinyllama"
OLLAMA_URL = "http://localhost:11434"
NO_ANSWER = "I couldn't find that in the documents you uploaded."

ANSWER_PROMPT = """You are DocuMind, an assistant that answers strictly from the provided document excerpts.

Rules:
- Use only the context below. Never invent facts.
- Cite the page you used inline, like [p.3].
- If the context does not answer the question, reply exactly: {fallback}
- Be direct and concise.

Context:
{context}

Conversation so far:
{history}

Question: {question}

Answer:"""

SUMMARY_PROMPT = """Read the excerpts and brief someone who has not seen the document.

Open with one sentence describing what the document covers, then list its key points as bullets.
Write about the subject matter only, never about these instructions.

Excerpts:
{context}

Briefing:"""

SUGGEST_PROMPT = """Based on the document excerpts below, write four specific questions a reader would likely ask.

Return only the questions, one per line, with no numbering or extra text.

Excerpts:
{context}

Questions:"""


def ollama_status(base_url: str = OLLAMA_URL) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/api/tags"
    request = urllib.request.Request(url, headers={"User-Agent": "DocuMind"})
    try:
        with urllib.request.urlopen(request, timeout=3) as response:
            payload = json.loads(response.read().decode())
        models = sorted(m.get("name", "") for m in payload.get("models", []) if m.get("name"))
        return {"running": True, "models": models, "error": None}
    except Exception as error:
        return {
            "running": False,
            "models": [],
            "error": f"Cannot reach Ollama at {base_url} ({error}). Run 'ollama serve'.",
        }


def similarity(distance: float) -> float:
    return round(max(0.0, min(1.0, 1.0 - (float(distance) ** 2) / 2.0)), 4)


class ChatMemory:
    def __init__(self, max_turns: int = 6):
        self.max_turns = max_turns
        self.messages: list[dict[str, str]] = []

    def add(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        limit = self.max_turns * 2
        if len(self.messages) > limit:
            self.messages = self.messages[-limit:]

    def as_text(self) -> str:
        if not self.messages:
            return "(no previous messages)"
        labels = {"user": "User", "assistant": "DocuMind"}
        return "\n".join(f"{labels[m['role']]}: {m['content']}" for m in self.messages)

    def clear(self) -> None:
        self.messages.clear()

    def as_markdown(self) -> str:
        lines = ["# DocuMind conversation", ""]
        for message in self.messages:
            speaker = "You" if message["role"] == "user" else "DocuMind"
            lines += [f"**{speaker}**", "", message["content"], ""]
        return "\n".join(lines)


class DocuMindEngine:
    def __init__(
        self,
        index: Optional[FAISS] = None,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.1,
        base_url: str = OLLAMA_URL,
    ):
        self.index = index
        self.model = model
        self.temperature = temperature
        self.base_url = base_url
        self.memory = ChatMemory()
        self._llm: Optional[ChatOllama] = None
        self._llm_key: tuple[str, float] = ("", -1.0)

    def set_index(self, index: Optional[FAISS]) -> None:
        self.index = index

    def configure(self, model: Optional[str] = None, temperature: Optional[float] = None) -> None:
        if model:
            self.model = model
        if temperature is not None:
            self.temperature = temperature

    @property
    def llm(self) -> ChatOllama:
        key = (self.model, self.temperature)
        if self._llm is None or self._llm_key != key:
            self._llm = ChatOllama(
                model=self.model,
                temperature=self.temperature,
                base_url=self.base_url,
            )
            self._llm_key = key
        return self._llm

    def search(self, question: str, k: int = 4, min_score: float = 0.0) -> list[dict[str, Any]]:
        if self.index is None:
            return []

        hits = []
        seen: set[tuple[str, int, int]] = set()
        for document, distance in self.index.similarity_search_with_score(question, k=k):
            meta = document.metadata or {}
            score = similarity(distance)
            if score < min_score:
                continue
            key = (meta.get("source", ""), meta.get("page", 0), meta.get("chunk", 0))
            if key in seen:
                continue
            seen.add(key)
            text = document.page_content.strip()
            hits.append({
                "source": meta.get("source", "unknown.pdf"),
                "page": meta.get("page", 1),
                "total_pages": meta.get("total_pages", 1),
                "score": score,
                "text": text,
                "snippet": text[:280] + ("..." if len(text) > 280 else ""),
            })
        return hits

    def _context(self, hits: list[dict[str, Any]]) -> str:
        return "\n\n".join(
            f"[{hit['source']} p.{hit['page']}]\n{hit['text']}" for hit in hits
        )

    def _guard(self) -> Optional[dict[str, Any]]:
        if self.index is None:
            return {
                "success": False,
                "answer": "No documents indexed yet. Upload a PDF to get started.",
                "sources": [],
            }
        status = ollama_status(self.base_url)
        if not status["running"]:
            return {"success": False, "answer": status["error"], "sources": []}
        return None

    def ask(self, question: str, k: int = 4) -> dict[str, Any]:
        blocked = self._guard()
        if blocked:
            return blocked

        started = time.time()
        hits = self.search(question, k=k)
        if not hits:
            return {"success": True, "answer": NO_ANSWER, "sources": [], "elapsed": 0.0}

        prompt = ANSWER_PROMPT.format(
            fallback=NO_ANSWER,
            context=self._context(hits),
            history=self.memory.as_text(),
            question=question,
        )

        try:
            answer = self._complete(prompt)
        except Exception as error:
            return {"success": False, "answer": f"Generation failed: {error}", "sources": []}

        self.memory.add("user", question)
        self.memory.add("assistant", answer)

        return {
            "success": True,
            "answer": answer,
            "sources": [{key: hit[key] for key in ("source", "page", "total_pages", "score", "snippet")} for hit in hits],
            "elapsed": round(time.time() - started, 2),
        }

    def ask_stream(self, question: str, k: int = 4) -> Iterator[dict[str, Any]]:
        blocked = self._guard()
        if blocked:
            yield {"type": "error", "message": blocked["answer"]}
            return

        started = time.time()
        hits = self.search(question, k=k)
        if not hits:
            yield {"type": "sources", "sources": []}
            yield {"type": "token", "text": NO_ANSWER}
            yield {"type": "done", "elapsed": 0.0}
            return

        yield {
            "type": "sources",
            "sources": [{key: hit[key] for key in ("source", "page", "total_pages", "score", "snippet")} for hit in hits],
        }

        prompt = ANSWER_PROMPT.format(
            fallback=NO_ANSWER,
            context=self._context(hits),
            history=self.memory.as_text(),
            question=question,
        )

        collected: list[str] = []
        try:
            for chunk in self.llm.stream(prompt):
                piece = getattr(chunk, "content", "") or ""
                if piece:
                    collected.append(piece)
                    yield {"type": "token", "text": piece}
        except Exception as error:
            yield {"type": "error", "message": f"Generation failed: {error}"}
            return

        answer = "".join(collected).strip()
        self.memory.add("user", question)
        self.memory.add("assistant", answer)
        yield {"type": "done", "elapsed": round(time.time() - started, 2)}

    def summarize(self, pages: list[dict[str, Any]], max_chars: int = 6000) -> str:
        excerpt = ""
        for page in pages:
            piece = f"[p.{page['page']}] {page['text']}\n\n"
            if len(excerpt) + len(piece) > max_chars:
                break
            excerpt += piece
        if not excerpt:
            return "There is no readable text in this document."
        return self._complete(SUMMARY_PROMPT.format(context=excerpt))

    def suggest(self, pages: list[dict[str, Any]], max_chars: int = 3000) -> list[str]:
        excerpt = ""
        for page in pages:
            piece = f"{page['text']}\n\n"
            if len(excerpt) + len(piece) > max_chars:
                break
            excerpt += piece
        if not excerpt:
            return []
        raw = self._complete(SUGGEST_PROMPT.format(context=excerpt))
        questions = []
        for line in raw.splitlines():
            line = line.strip().lstrip("-*0123456789. ").strip()
            if len(line) > 10:
                questions.append(line)
        return questions[:4]

    def _complete(self, prompt: str) -> str:
        response = self.llm.invoke(prompt)
        text = getattr(response, "content", response)
        return str(text).strip()

    def clear_history(self) -> None:
        self.memory.clear()

