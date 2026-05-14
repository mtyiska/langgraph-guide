import os
import json
from typing import Optional, Literal, Union
from pydantic import BaseModel, field_validator, model_validator


class ToolSuccess(BaseModel):
    status: Literal["success"] = "success"
    result: str
    metadata: dict = {}


class ToolError(BaseModel):
    status: Literal["error"] = "error"
    error_type: Literal[
        "not_found", "permission_denied", "timeout",
        "invalid_input", "unknown"
    ]
    message: str
    recoverable: bool
    suggestion: str


ToolResponse = Union[ToolSuccess, ToolError]


def parse_tool_response(raw: str) -> ToolResponse:
    """Parse a tool's JSON response into a typed ToolResponse."""
    try:
        data = json.loads(raw)
        if data.get("status") == "success":
            return ToolSuccess.model_validate(data)
        else:
            return ToolError.model_validate(data)
    except Exception:
        # Raw string response from a tool that doesn't use the schema
        return ToolSuccess(result=raw)


class Citation(BaseModel):
    source_path: str
    excerpt: str
    relevance: str

    @field_validator("source_path")
    @classmethod
    def source_must_exist(cls, v: str) -> str:
        # Allow relative paths — only validate if the path is absolute or clearly a file
        if os.path.isabs(v) and not os.path.exists(v):
            raise ValueError(f"Cited source does not exist: {v}")
        return v

    @field_validator("excerpt")
    @classmethod
    def excerpt_not_empty(cls, v: str) -> str:
        if len(v.strip()) < 5:
            raise ValueError("Citation excerpt too short to be meaningful")
        return v


class ResearchAnswer(BaseModel):
    answer: str
    confidence: Literal["high", "medium", "low"]
    citations: list[Citation] = []
    gaps: list[str] = []
    answer_complete: bool = True

    @field_validator("answer")
    @classmethod
    def answer_not_empty(cls, v: str) -> str:
        if len(v.strip()) < 20:
            raise ValueError(
                "Answer too short — agent may have failed to produce content"
            )
        return v

    @model_validator(mode="after")
    def citations_required_for_high_confidence(self) -> "ResearchAnswer":
        if self.confidence == "high" and len(self.citations) == 0:
            raise ValueError(
                "High confidence answer requires at least one citation"
            )
        return self


def extract_json_from_text(text: str) -> str:
    """
    Extract a JSON object from text that may include surrounding prose
    or markdown code fences.
    """
    text = text.strip()

    # Strip markdown fences
    if text.startswith("```"):
        lines = text.splitlines()
        inner = []
        in_fence = False
        for line in lines:
            if line.strip().startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence or not text.startswith("```"):
                inner.append(line)
        text = "\n".join(inner).strip()

    # Find first { and last } to extract JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]

    return text