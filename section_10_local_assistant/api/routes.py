import uuid
import json
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from agent.graph import build_graph
from agent.state import make_initial_state
from infrastructure.trace_store import TraceStore
from infrastructure.database import init_all_databases
from api.session import SessionManager
from api.models import ChatRequest, ChatResponse, ApprovalRequest
from config import BASE_DIR

logger = logging.getLogger(__name__)

app          = FastAPI(title="Local Assistant")
trace_store  = TraceStore()
session_mgr  = SessionManager()
graph, run_context = build_graph(trace_store)


@app.on_event("startup")
def startup():
    init_all_databases()
    logger.info("Local Assistant started.")


@app.get("/", response_class=HTMLResponse)
def frontend():
    html_path = BASE_DIR / "frontend" / "index.html"
    return HTMLResponse(content=html_path.read_text())


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    session_id = session_mgr.get_or_create_session(request.session_id)
    thread_id  = session_mgr.session_to_thread_id(session_id)
    run_id     = str(uuid.uuid4())

    run_context["run_id"]       = run_id
    run_context["thread_id"]    = thread_id
    run_context["node_index"]   = 0
    run_context["trajectory"]   = []
    run_context["total_tokens"] = 0

    config = {"configurable": {"thread_id": thread_id}}

    try:
        initial = make_initial_state(
            user_message=request.message,
            session_id=session_id,
            thread_id=thread_id,
            run_id=run_id,
        )
        result = graph.invoke(initial, config)
    except Exception as e:
        logger.error(f"Graph invocation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    # Check if paused for HITL
    current_state = graph.get_state(config)
    pending = None
    if current_state.next:
        pending = result.get("pending_write_operation")

    # Save run trace
    from infrastructure.trace_store import RunTrace
    from datetime import datetime
    rt = RunTrace(
        run_id=run_id,
        thread_id=thread_id,
        user_query=request.message,
        trajectory=run_context["trajectory"],
        total_tokens=run_context.get("total_tokens", 0),
        final_status=result.get("status", "complete"),
        output_valid=result.get("output_valid", False),
        guardrail_triggered=result.get("guardrail_triggered", False),
        loop_detected=result.get("loop_detected", False),
        degraded=result.get("status") == "degraded",
        final_answer_preview=(result.get("final_response") or "")[:500],
    )
    rt.end_time = datetime.utcnow()
    trace_store.save_run_trace(rt)

    return ChatResponse(
        response=result.get("final_response", "No response generated."),
        session_id=session_id,
        run_id=run_id,
        status=result.get("status", "complete"),
        sources=result.get("sources_used", []),
        pending_approval=pending,
    )


@app.post("/approve")
def approve_action(request: ApprovalRequest):
    thread_id = session_mgr.session_to_thread_id(request.session_id)
    config    = {"configurable": {"thread_id": thread_id}}

    graph.update_state(config, {
        "write_approved":          request.approved,
        "write_decision_note":     request.note,
        "pending_write_operation": None,
    })

    result = graph.invoke(None, config)
    return {
        "response": result.get("final_response", "Action processed."),
        "status":   result.get("status", "complete"),
    }


@app.get("/sessions/{session_id}/tasks")
def get_tasks(session_id: str):
    from agent.tools.task_tools import list_tasks
    result = list_tasks.invoke({"status": None, "priority": None})
    return json.loads(result)


@app.get("/api/runs")
def get_runs(limit: int = 100):
    return trace_store.get_runs(limit=limit)


@app.get("/api/runs/{run_id}/traces")
def get_node_traces(run_id: str):
    return trace_store.get_node_traces(run_id)


@app.get("/api/token-stats")
def token_stats():
    return trace_store.get_token_stats()


@app.get("/health")
def health():
    return {"status": "ok"}