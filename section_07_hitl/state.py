from typing import TypedDict, Annotated, Optional, Literal
from operator import add


class ProposedChange(TypedDict):
    file_path: str
    line_start: int
    line_end: int
    original_code: str
    proposed_code: str
    reason: str
    issue_type: Literal["bug", "type_error", "style", "performance", "security"]
    severity: Literal["critical", "high", "medium", "low"]


class ReviewDecision(TypedDict):
    change: ProposedChange
    decision: Literal["approved", "rejected", "edited"]
    edited_code: Optional[str]
    reviewer_note: Optional[str]


class CodeReviewState(TypedDict):
    # Input
    files_to_review: list[str]

    # Analysis
    files_read: Annotated[list[dict], add]         # {path, content, line_count}
    issues_found: Annotated[list[dict], add]        # raw analysis per file
    proposed_changes: Annotated[list[ProposedChange], add]

    # Human review
    pending_change: Optional[ProposedChange]
    formatted_diff: Optional[str]
    review_decisions: Annotated[list[ReviewDecision], add]

    # Execution
    changes_applied: Annotated[list[str], add]
    current_file_index: int
    current_change_index: int

    # Control
    status: Literal["analysing", "reviewing", "applying", "complete", "failed"]
    error: Optional[str]
    session_id: str