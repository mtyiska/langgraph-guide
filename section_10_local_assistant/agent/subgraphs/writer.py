import logging
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from agent.state import AssistantState
from config import PRIMARY_MODEL, OLLAMA_BASE_URL, PROMPTS_DIR

logger = logging.getLogger(__name__)


def writer_node(state: AssistantState) -> dict:
    llm = ChatOllama(model=PRIMARY_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.3)

    system       = (PROMPTS_DIR / "writer.txt").read_text()
    instructions = state.get("supervisor_instructions", "Complete the writing task.")

    worker_results = state.get("worker_results", [])
    research_context = ""
    if worker_results:
        research_context = "\n\n".join(
            f"Research from {r['worker']}:\n{r['result']}"
            for r in worker_results if r["worker"] == "researcher"
        )

    prompt = instructions
    if research_context:
        prompt = f"Source material:\n{research_context}\n\nTask: {instructions}"

    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=prompt),
    ])

    return {
        "messages":       [response],
        "worker_results": [{"worker": "writer", "result": response.content[:400]}],
        "_llm_response":  response,
    }


def build_writer_subgraph():
    builder = StateGraph(AssistantState)
    builder.add_node("writer", writer_node)
    builder.add_edge(START, "writer")
    builder.add_edge("writer", END)
    return builder.compile()