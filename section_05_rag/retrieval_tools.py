import sys
sys.path.append("..")

import os
from typing import Optional
from langchain_core.tools import tool
from ingestion import get_collection, get_source_chunk_counts


def build_retrieval_tools(collection=None):
    if collection is None:
        collection = get_collection()

    @tool
    def list_document_sources() -> str:
        """
        List all available document sources in the collection with their chunk counts.
        Call this first when you don't know what documents are available.
        Returns filenames and the number of searchable chunks per file.
        """
        try:
            counts = get_source_chunk_counts(collection)
            if not counts:
                return "No documents found in the collection. Run ingestion first."
            lines = ["Available documents:"]
            for source, count in sorted(counts.items()):
                lines.append(f"  - {source} ({count} chunks)")
            return "\n".join(lines)
        except Exception as e:
            return f"Error listing sources: {e}"

    @tool
    def search_documents(
        query: str,
        n_results: int = 5,
        source_filter: Optional[str] = None
    ) -> str:
        """
        Search the document collection for chunks relevant to the query.
        Returns matching text chunks with source filenames and relevance scores.

        Args:
            query: The search query. Be specific. If initial results are poor,
                   try synonyms, related terms, or more specific language.
            n_results: Number of results (default 5, max 10).
            source_filter: Optional filename to restrict search to one document.
                           Use after list_document_sources identifies the right file.

        Relevance scores: < 0.5 = high, 0.5-1.0 = medium, > 1.0 = low.
        Call multiple times with refined queries if initial results are insufficient.
        """
        try:
            where = {"source": source_filter} if source_filter else None
            results = collection.query(
                query_texts=[query],
                n_results=min(int(n_results), 10),
                where=where,
                include=["documents", "metadatas", "distances"]
            )

            if not results["documents"][0]:
                return f"No results found for '{query}'. Try different search terms."

            formatted = []
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ):
                relevance = "high" if dist < 0.5 else "medium" if dist < 1.0 else "low"
                formatted.append(
                    f"[Source: {meta['source']} | "
                    f"Section: {meta.get('header_path', 'unknown')} | "
                    f"Relevance: {relevance} (score: {dist:.3f})]\n{doc}"
                )

            return "\n\n---\n\n".join(formatted)

        except Exception as e:
            return f"Error searching documents: {e}"

    @tool
    def get_document_summary(source: str) -> str:
        """
        Retrieve the first two chunks of a specific document as an overview.
        Use this to understand whether a document is worth searching in depth
        before committing multiple search calls to it.

        Args:
            source: The filename of the document (e.g. 'budget_overview.md').
                    Use list_document_sources to see available filenames.
        """
        try:
            results = collection.query(
                query_texts=["introduction overview summary"],
                n_results=2,
                where={"source": source},
                include=["documents", "metadatas"]
            )

            if not results["documents"][0]:
                return f"Document '{source}' not found. Use list_document_sources to see available files."

            parts = [f"Overview of '{source}':"]
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                parts.append(f"[Section: {meta.get('header_path', 'root')}]\n{doc}")

            return "\n\n".join(parts)

        except Exception as e:
            return f"Error getting document summary: {e}"

    return [list_document_sources, search_documents, get_document_summary]