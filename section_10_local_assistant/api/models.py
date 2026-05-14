from pydantic import BaseModel
from typing import Optional


class ChatRequest(BaseModel):
    message:    str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response:         str
    session_id:       str
    run_id:           str
    status:           str
    sources:          list[str] = []
    pending_approval: Optional[dict] = None


class ApprovalRequest(BaseModel):
    session_id:    str
    approved:      bool
    note:          Optional[str] = None
    edited_value:  Optional[str] = None