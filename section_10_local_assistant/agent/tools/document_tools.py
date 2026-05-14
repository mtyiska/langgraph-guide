import logging
from langchain_core.tools import tool
from infrastructure.vector_store import VectorStoreWrapper
from agent.tools.base import retry_with_backoff, ToolSuccess, ToolError
from config import KNOWLEDGE_BASE_DIR

logger = logging.getLogger(__name__)
_vector_store = None


def get_vector_store() -> VectorStoreWrapper:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStoreWrapper()
    return _vector_store


@tool
def search_documents(query: str, max_results: int = 5) -> str:
    """
    Search the knowledge base for documents relevant to the query.
    Returns matching excerpts with source file paths.
    Use for factual questions, research tasks, or when you need
    information from the user's notes and documents.
    """
    try:
        results = get_vector_store().search(query, k=max_results)
        if not results:
            return ToolError(
                error_type="not_found",
                message=f"No documents found for: {query}",
                recoverable=True,
                suggestion="Try broader or different search terms",
            ).model_dump_json()

        formatted = []
        for doc, score in results:
            source = doc.metadata.get("filename", doc.metadata.get("source", "unknown"))
            formatted.append(
                f"Source: {source}\nRelevance: {score:.2f}\nContent: {doc.page_content[:600]}"
            )

        return ToolSuccess(
            result="\n\n---\n\n".join(formatted),
            metadata={"result_count": len(results)},
        ).model_dump_json()

    except Exception as e:
        logger.error(f"search_documents failed: {e}")
        return ToolError(
            error_type="unknown",
            message=str(e),
            recoverable=False,
            suggestion="Knowledge base may be unavailable.",
        ).model_dump_json()


@tool
def list_documents() -> str:
    """
    List all documents currently in the knowledge base.
    Returns filenames and sizes.
    Use when the user asks what documents are available.
    """
    files = [
        f for f in KNOWLEDGE_BASE_DIR.rglob("*")
        if f.is_file() and f.suffix in {".md", ".txt", ".pdf"}
    ]
    if not files:
        return ToolSuccess(
            result="No documents in knowledge base. Add files to data/knowledge_base/",
            metadata={"count": 0},
        ).model_dump_json()

    listing = [f"- {f.name} ({f.stat().st_size // 1024}KB)" for f in sorted(files)]
    return ToolSuccess(
        result="\n".join(listing),
        metadata={"count": len(files)},
    ).model_dump_json()


@tool
def index_knowledge_base() -> str:
    """
    Re-index all documents in the knowledge base.
    Use when new documents have been added.
    """
    try:
        vs = get_vector_store()
        vs.index_directory(KNOWLEDGE_BASE_DIR)
        count = vs.count()
        return ToolSuccess(
            result=f"Knowledge base indexed. {count} chunks available.",
            metadata={"chunk_count": count},
        ).model_dump_json()
    except Exception as e:
        return ToolError(
            error_type="unknown",
            message=str(e),
            recoverable=False,
            suggestion="Check Ollama is running and the embedding model is available.",
        ).model_dump_json()


DOCUMENT_TOOLS = [search_documents, list_documents, index_knowledge_base]