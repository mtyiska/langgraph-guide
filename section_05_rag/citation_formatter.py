from typing import Optional


def format_citations(sources_used: list[dict]) -> str:
    if not sources_used:
        return ""

    seen = set()
    deduped = []
    for s in sources_used:
        key = f"{s.get('source')}:{s.get('chunk_preview', '')[:50]}"
        if key not in seen:
            seen.add(key)
            deduped.append(s)

    lines = ["\n\n---\n**Sources:**"]
    for i, s in enumerate(deduped, 1):
        source = s.get("source", "unknown")
        section = s.get("section", "")
        query = s.get("query_used", "")
        relevance = s.get("relevance_score", None)
        preview = s.get("chunk_preview", "")[:120]

        line = f"{i}. `{source}`"
        if section and section != "unknown":
            line += f" › {section}"
        if relevance is not None:
            line += f" (relevance score: {relevance:.3f})"
        if preview:
            line += f"\n   > {preview}..."
        lines.append(line)

    return "\n".join(lines)


def build_citation_record(
    source: str,
    section: str,
    chunk_text: str,
    distance: float,
    query_used: str
) -> dict:
    return {
        "source": source,
        "section": section,
        "chunk_preview": chunk_text[:200],
        "relevance_score": distance,
        "query_used": query_used,
    }


def parse_search_results_for_citations(raw_result: str, query_used: str) -> list[dict]:
    citations = []
    blocks = raw_result.split("\n\n---\n\n")
    for block in blocks:
        lines = block.strip().split("\n")
        if not lines:
            continue
        header = lines[0]
        body = "\n".join(lines[1:]).strip()

        source = "unknown"
        section = "unknown"
        distance = 1.0

        if header.startswith("["):
            parts = header.strip("[]").split("|")
            for part in parts:
                part = part.strip()
                if part.startswith("Source:"):
                    source = part.replace("Source:", "").strip()
                elif part.startswith("Section:"):
                    section = part.replace("Section:", "").strip()
                elif "score:" in part:
                    try:
                        distance = float(part.split("score:")[-1].strip().rstrip("]"))
                    except ValueError:
                        pass

        if body:
            citations.append(build_citation_record(source, section, body, distance, query_used))

    return citations