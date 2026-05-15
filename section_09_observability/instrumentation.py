import time
import uuid
import logging
from datetime import datetime
from functools import wraps
from typing import Callable

from langchain_core.messages import AIMessage
from trace_schema import NodeTrace
from trace_store import TraceStore

logger = logging.getLogger(__name__)


def _extract_tokens(last_msg: AIMessage) -> tuple[int | None, int | None, str | None]:
    """
    Extract (input_tokens, output_tokens, model_name) from an AIMessage.
    Handles multiple metadata layouts returned by different Ollama versions.
    """
    meta = getattr(last_msg, "response_metadata", {}) or {}

    # Layout 1: top-level prompt_eval_count / eval_count (most Ollama versions)
    input_tok = meta.get("prompt_eval_count")
    output_tok = meta.get("eval_count")

    # Layout 2: nested under "usage"
    if input_tok is None:
        usage = meta.get("usage") or {}
        input_tok = (
            usage.get("prompt_tokens")
            or usage.get("input_tokens")
            or usage.get("prompt_token_count")
        )
        output_tok = (
            usage.get("completion_tokens")
            or usage.get("output_tokens")
            or usage.get("candidates_token_count")
        )

    # Layout 3: usage_metadata attribute on the message itself
    if input_tok is None:
        usage_meta = getattr(last_msg, "usage_metadata", {}) or {}
        input_tok = usage_meta.get("input_tokens")
        output_tok = usage_meta.get("output_tokens")

    model_name = (
        meta.get("model")
        or meta.get("model_name")
        or meta.get("model_id")
    )

    return input_tok, output_tok, model_name


def traced_node(node_func: Callable, trace_store: TraceStore, run_context: dict) -> Callable:
    """
    Wraps a node function to automatically record a NodeTrace.

    run_context is a mutable dict shared across all nodes in one run:
      {run_id, thread_id, node_index (int), trajectory (list), total_tokens (int)}
    """

    @wraps(node_func)
    def wrapper(state: dict) -> dict:
        trace = NodeTrace(
            run_id=run_context["run_id"],
            thread_id=run_context.get("thread_id", ""),
            node_name=node_func.__name__,
            node_index=run_context["node_index"],
            start_time=datetime.utcnow(),
            input_state_keys=list(state.keys()),
            input_snapshot=dict(state),
        )
        run_context["node_index"] += 1

        start = time.perf_counter()
        result = None

        try:
            result = node_func(state)
            trace.status = "success"
            trace.output_state_keys = list(result.keys()) if result else []
            trace.output_snapshot = dict(result) if result else {}

            # Extract token usage from the last AIMessage if this was an LLM node
            messages = (result or {}).get("messages", [])
            if messages:
                last = messages[-1] if isinstance(messages, list) else None
                if last and isinstance(last, AIMessage):
                    input_tok, output_tok, model_name = _extract_tokens(last)

                    trace.input_tokens = input_tok
                    trace.output_tokens = output_tok
                    trace.model_name = model_name

                    if input_tok and output_tok:
                        trace.total_tokens = input_tok + output_tok
                    elif input_tok:
                        trace.total_tokens = input_tok
                    elif output_tok:
                        trace.total_tokens = output_tok

                    # Debug log so you can see what metadata arrived
                    logger.debug(
                        f"{node_func.__name__} tokens: "
                        f"in={input_tok} out={output_tok} model={model_name} "
                        f"raw_meta={getattr(last, 'response_metadata', {})}"
                    )

            # Accumulate tokens in run context
            if trace.total_tokens:
                run_context["total_tokens"] = (
                    run_context.get("total_tokens", 0) + trace.total_tokens
                )

            # Extract tool calls from the last AIMessage
            if messages and isinstance(messages[-1], AIMessage):
                tcs = getattr(messages[-1], "tool_calls", []) or []
                trace.tool_calls = [
                    {"name": tc.get("name"), "args": tc.get("args")} for tc in tcs
                ]

            return result

        except Exception as e:
            trace.status = "error"
            trace.error_message = str(e)
            raise

        finally:
            trace.end_time = datetime.utcnow()
            trace.duration_ms = (time.perf_counter() - start) * 1000
            trace_store.save_node_trace(trace)
            run_context["trajectory"].append(node_func.__name__)

    return wrapper