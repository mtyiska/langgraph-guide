import sys
sys.path.append("..")

import os
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

SAMPLE_DOCS = {
    "project_atlas.md": """# Project Atlas

## Overview

Project Atlas is a data platform initiative aimed at centralising all internal reporting.
The platform will serve 200 internal users across finance, operations, and product teams.
The project began in Q1 2024 with an initial budget of $320,000.

## Goals

The primary goal is to reduce report generation time from 3 days to under 4 hours.
Secondary goals include improving data accuracy and enabling self-serve analytics.
The platform will integrate with existing ERP and CRM systems.

## Timeline

Phase 1 runs from January to June 2024 and covers data ingestion and core API development.
Phase 2 runs from July to December 2024 and covers the user-facing dashboard and reporting layer.
A beta launch is planned for August 15, 2024 with full launch on November 1, 2024.
""",

    "budget_overview.md": """# Budget Overview

## Total Allocation

The total approved budget for fiscal year 2024 is $320,000.
This covers all phases of development, infrastructure, and licensing costs.

## Personnel

Engineering costs for 4 FTEs over 12 months total $204,000.
Design costs for 1 FTE over 6 months total $36,000.
QA costs for 1 FTE over 8 months total $44,000.
Project management at 0.5 FTE over 12 months totals $42,000.
Total personnel costs: $326,000 — note this exceeds the total budget due to phased hiring.

## Infrastructure

Cloud infrastructure estimated at $18,000 for the full year.
Third-party data licensing costs $24,000 annually.
SaaS tooling including monitoring and CI/CD tools costs $8,000.
Total infrastructure: $50,000.

## Contingency

A contingency reserve of $20,000 is held for unplanned expenses.
This reserve requires VP approval to access.

## Current Spend

As of March 31, 2024: $87,000 spent, representing 27% of total budget.
Remaining budget: $233,000.
Budget health status: ON TRACK.
""",

    "technical_decisions.md": """# Technical Decisions

## Architecture Decision Record 001 — Database Selection

Date: January 15, 2024
Decision: PostgreSQL over MySQL or MongoDB.
Rationale: Strong JSON support, better analytical query performance, team familiarity.
Consequences: Requires PostgreSQL expertise in hiring. No document flexibility.

## Architecture Decision Record 002 — API Framework

Date: January 22, 2024
Decision: FastAPI over Django REST Framework.
Rationale: Native async support, automatic OpenAPI documentation, better performance.
Consequences: Smaller ecosystem than Django. Less built-in admin tooling.

## Architecture Decision Record 003 — Frontend Framework

Date: February 1, 2024
Decision: React with TypeScript over Vue or Angular.
Rationale: Largest talent pool, strong typing reduces runtime errors, existing component library.
Consequences: Higher initial complexity. Requires TypeScript expertise.

## Architecture Decision Record 004 — Deployment

Date: February 10, 2024
Decision: AWS ECS with Fargate over self-managed Kubernetes.
Rationale: Reduced operational overhead, no cluster management, scales to zero.
Consequences: AWS vendor lock-in. Higher per-unit cost at scale.
""",
}

header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
)

for size in [500, 1500, 3000]:
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " "]
    )

    total_chunks = 0
    print(f"\n=== chunk_size={size} ===")
    for filename, content in SAMPLE_DOCS.items():
        header_chunks = header_splitter.split_text(content)
        final_chunks = char_splitter.split_documents(header_chunks)
        total_chunks += len(final_chunks)
        print(f"  {filename}: {len(final_chunks)} chunks")

        if size == 1500:
            for i, chunk in enumerate(final_chunks):
                header_path = " > ".join(filter(None, [
                    chunk.metadata.get("h1"),
                    chunk.metadata.get("h2"),
                    chunk.metadata.get("h3"),
                ]))
                print(f"    [{i}] headers='{header_path}' chars={len(chunk.page_content)}")
                print(f"         preview: {chunk.page_content[:80].strip()}...")

    print(f"  TOTAL chunks across all docs: {total_chunks}")

print("\n=== Smallest and Largest Chunks (chunk_size=1500) ===")
char_splitter_default = RecursiveCharacterTextSplitter(
    chunk_size=1500, chunk_overlap=200, separators=["\n\n", "\n", ". ", " "]
)
all_chunks = []
for filename, content in SAMPLE_DOCS.items():
    header_chunks = header_splitter.split_text(content)
    final_chunks = char_splitter_default.split_documents(header_chunks)
    for chunk in final_chunks:
        chunk.metadata["source"] = filename
    all_chunks.extend(final_chunks)

sorted_chunks = sorted(all_chunks, key=lambda c: len(c.page_content))
print(f"Smallest chunk ({len(sorted_chunks[0].page_content)} chars):")
print(f"  {sorted_chunks[0].page_content[:150]}")
print(f"Largest chunk ({len(sorted_chunks[-1].page_content)} chars):")
print(f"  {sorted_chunks[-1].page_content[:150]}...")