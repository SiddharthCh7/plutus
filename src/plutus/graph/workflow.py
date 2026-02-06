"""LangGraph workflow construction.

Builds the multi-agent graph based on task type.
Uses supervisor pattern with coordinator routing to specialists.
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import StateGraph, END

from plutus.graph.state import PlutusState
from plutus.graph.nodes import (
    coordinator_node,
    portfolio_tracker_node,
    news_monitor_node,
    market_analyst_node,
    risk_assessor_node,
    opportunity_scout_node,
    trend_interpreter_node,
    report_generator_node,
    notifier_node,
)
from plutus.logging import get_logger

logger = get_logger(__name__)


def should_continue(state: PlutusState) -> Literal["continue", "report"]:
    """Decide if workflow should continue or generate report."""
    # Check if all requested agents have run
    task_agents = get_agents_for_task(state.task_type)
    completed = set(state.agents_invoked)
    required = set(task_agents)
    
    if required.issubset(completed):
        return "report"
    return "continue"


def get_agents_for_task(task_type: str) -> list[str]:
    """Get list of agents for a task type."""
    task_agents = {
        "portfolio_monitor": [
            "portfolio_tracker",
            "news_monitor",
            "risk_assessor",
        ],
        "opportunity_discovery": [
            "news_monitor",
            "market_analyst",
            "opportunity_scout",
            "trend_interpreter",
        ],
        "news_digest": [
            "news_monitor",
        ],
    }
    return task_agents.get(task_type, ["news_monitor"])


def route_to_next_agent(state: PlutusState) -> str:
    """Route to next agent based on task requirements.
    
    Coordinator determines which agent runs next.
    """
    task_agents = get_agents_for_task(state.task_type)
    completed = set(state.agents_invoked)
    
    for agent in task_agents:
        if agent not in completed:
            logger.debug("Routing to agent", agent=agent)
            return agent
    
    return "report_generator"


def create_workflow() -> StateGraph:
    """Create the Plutus multi-agent workflow.
    
    Architecture:
    - Coordinator routes to specialist agents
    - Each task type invokes specific agents
    - All paths converge to report generation
    """
    # Create the graph
    workflow = StateGraph(PlutusState)
    
    # Add nodes
    workflow.add_node("coordinator", coordinator_node)
    workflow.add_node("portfolio_tracker", portfolio_tracker_node)
    workflow.add_node("news_monitor", news_monitor_node)
    workflow.add_node("market_analyst", market_analyst_node)
    workflow.add_node("risk_assessor", risk_assessor_node)
    workflow.add_node("opportunity_scout", opportunity_scout_node)
    workflow.add_node("trend_interpreter", trend_interpreter_node)
    workflow.add_node("report_generator", report_generator_node)
    workflow.add_node("notifier", notifier_node)
    
    # Set entry point
    workflow.set_entry_point("coordinator")
    
    # Add conditional edges from coordinator
    workflow.add_conditional_edges(
        "coordinator",
        route_to_next_agent,
        {
            "portfolio_tracker": "portfolio_tracker",
            "news_monitor": "news_monitor",
            "market_analyst": "market_analyst",
            "risk_assessor": "risk_assessor",
            "opportunity_scout": "opportunity_scout",
            "trend_interpreter": "trend_interpreter",
            "report_generator": "report_generator",
        },
    )
    
    # Each specialist returns to coordinator
    for agent in [
        "portfolio_tracker",
        "news_monitor",
        "market_analyst",
        "risk_assessor",
        "opportunity_scout",
        "trend_interpreter",
    ]:
        workflow.add_edge(agent, "coordinator")
    
    # Report generation flow
    workflow.add_conditional_edges(
        "report_generator",
        lambda s: "notify" if s.should_notify else "end",
        {
            "notify": "notifier",
            "end": END,
        },
    )
    workflow.add_edge("notifier", END)
    
    logger.info("Created Plutus workflow graph")
    
    return workflow


# Compiled workflow singleton
_compiled_workflow = None


def get_workflow():
    """Get the compiled workflow (singleton)."""
    global _compiled_workflow
    if _compiled_workflow is None:
        graph = create_workflow()
        _compiled_workflow = graph.compile()
        logger.info("Compiled Plutus workflow")
    return _compiled_workflow
