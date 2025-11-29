from typing import Dict, Any

from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda

from src.mcp_research_assistant import MCPResearchAssistant


class GraphState(dict):
    """Simple dict-based state for LangGraph."""
    pass


def build_graph(assistant: MCPResearchAssistant) -> StateGraph:
    graph = StateGraph(GraphState)

    async def retrieve_node(state: GraphState) -> GraphState:
        query = state["query"]
        papers = await assistant._retrieve_papers(
            query, max_papers=state.get("max_papers", 10), sources=["arxiv"], context_id="graph"
        )
        state["papers"] = papers
        return state

    async def summarize_node(state: GraphState) -> GraphState:
        query = state["query"]
        papers = state["papers"]
        summary, meta = await assistant._generate_summary(query, papers, context_id="graph")
        state["summary"] = summary
        state["summary_meta"] = meta
        return state

    async def verify_node(state: GraphState) -> GraphState:
        query = state["query"]
        papers = state["papers"]
        summary = state["summary"]
        verification = await assistant._verify_summary(query, summary, papers, context_id="graph")
        state["verification"] = verification
        return state

    graph.add_node("retrieve", RunnableLambda(retrieve_node))
    graph.add_node("summarize", RunnableLambda(summarize_node))
    graph.add_node("verify", RunnableLambda(verify_node))

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "summarize")
    graph.add_edge("summarize", "verify")
    graph.add_edge("verify", END)

    return graph


async def run_graph(assistant: MCPResearchAssistant, query: str) -> Dict[str, Any]:
    g = build_graph(assistant).compile()
    out = await g.ainvoke({"query": query, "max_papers": 8})
    # Format like the orchestrator
    verification = out.get("verification", {})
    summary = out.get("summary", "")
    papers = out.get("papers", [])
    return assistant._format_response(query, summary, papers, verification)
