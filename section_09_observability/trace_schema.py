from dataclasses import dataclass, field
from typing import Optional, Any
from datetime import datetime
import uuid


@dataclass
class NodeTrace:
    # Identity
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str = ""
    thread_id: str = ""

    # Node identification
    node_name: str = ""
    node_index: int = 0

    # Timing
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None

    # Inputs and outputs
    input_state_keys: list = field(default_factory=list)
    output_state_keys: list = field(default_factory=list)
    input_snapshot: dict = field(default_factory=dict)
    output_snapshot: dict = field(default_factory=dict)

    # LLM usage
    model_name: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

    # Tool calls
    tool_calls: list = field(default_factory=list)

    # Status
    status: str = "pending"
    error_message: Optional[str] = None
    retry_count: int = 0


@dataclass
class RunTrace:
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    thread_id: str = ""
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None

    user_query: str = ""
    trajectory: list = field(default_factory=list)

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0

    final_status: str = ""
    output_valid: bool = False
    guardrail_triggered: bool = False
    loop_detected: bool = False
    degraded: bool = False

    final_answer_preview: str = ""