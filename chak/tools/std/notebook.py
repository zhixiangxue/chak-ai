"""
Notebook: Built-in persistent memory tool for chak

Wraps seeka (https://github.com/nicepkg/seeka) as an LLM-friendly notebook
that agents can use to note down and recall information across conversations.

Design philosophy:
  - Only two verbs: note (write) and recall (search)
  - No metadata, no filter, no CRUD — the LLM should never hesitate
  - ``note()`` internally calls seeka's ``remember()`` (note + dream in one shot),
    so every note is immediately searchable

Usage::

    from chak.tools.std import Notebook, NotebookBackend

    nb = Notebook(
        path="./agent_notebook",
        embedding_uri="bailian/text-embedding-v3",
        embedding_api_key=os.environ["BAILIAN_API_KEY"],
        llm_uri="bailian/qwen-plus",
        llm_api_key=os.environ["BAILIAN_API_KEY"],
    )

    conv = Conversation(model, tools=[nb])
"""

from typing import Optional

try:
    from seeka import StorageBackend as NotebookBackend
except ImportError:
    raise ImportError(
        "The Notebook tool requires the seeka package. "
        "Install it with: pip install seeka"
    )


class Notebook:
    """A persistent, searchable notebook for LLM agents.

    Powered by seeka under the hood — but you never need to import it.

    Parameters
    ----------
    path:
        Directory for persistent storage (vector DB + SQLite).
        Created automatically if it does not exist.
    namespace:
        Logical namespace for multi-tenant isolation.  Defaults to ``"default"``.
    storage:
        Vector backend.  Defaults to :attr:`NotebookBackend.LANCEDB`.
    embedding_uri:
        Embedding model URI in chak format (e.g. ``"openai/text-embedding-3-small"``).
        When ``None``, falls back to a local sentence-transformers model.
    embedding_api_key:
        API key for the embedding provider.  Required when *embedding_uri* is set.
    llm_uri:
        LLM model URI for memory extraction / conflict resolution during
        ``note()``.  When ``None``, notes are stored without LLM refinement.
    llm_api_key:
        API key for the LLM provider.  Required when *llm_uri* is set.
    rerank_uri:
        Re-ranker model URI for improving recall precision.  Optional.
    rerank_api_key:
        API key for the re-ranker provider.  Required when *rerank_uri* is set.
    skills:
        List of seeka skill directory paths for customising memory extraction.
    """

    def __init__(
        self,
        path: str,
        namespace: str = "default",
        storage: NotebookBackend = NotebookBackend.LANCEDB,
        embedding_uri: Optional[str] = None,
        embedding_api_key: Optional[str] = None,
        llm_uri: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        rerank_uri: Optional[str] = None,
        rerank_api_key: Optional[str] = None,
        skills: Optional[list[str]] = None,
    ):
        from seeka import Memory as SeekaMemory

        self._mem = SeekaMemory(
            path=path,
            namespace=namespace,
            storage=storage,
            embedding_uri=embedding_uri,
            embedding_api_key=embedding_api_key,
            llm_uri=llm_uri,
            llm_api_key=llm_api_key,
            rerank_uri=rerank_uri,
            rerank_api_key=rerank_api_key,
            skills=skills,
        )

    # ------------------------------------------------------------------
    # Public API — only these two methods are exposed to the LLM
    # ------------------------------------------------------------------

    async def note(self, content: str) -> str:
        """Write down information worth keeping in your notebook.

        Call this when the user shares something you may need to recall later —
        preferences, facts, decisions, context, etc.
        """
        memos = await self._mem.remember(content, metadata={})
        if not memos:
            return "Noted (no new memories extracted)."
        lines = [f"- {m.content}" for m in memos]
        return "Noted:\n" + "\n".join(lines)

    async def recall(self, query: str, n: int = 5) -> str:
        """Search the notebook for previously noted information.

        Args:
            query: What you are looking for (natural language).
            n: Maximum number of results to return (default 5).
        """
        memos = await self._mem.recall(query, n=n)
        if not memos:
            return "Nothing found in the notebook for that query."
        lines = [f"[{m.id[:8]}] {m.content}" for m in memos]
        return "\n".join(lines)
