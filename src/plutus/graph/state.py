"""Shared state for LangGraph workflow.

This is the central state that flows through all agents.
Designed for context efficiency - only essential data in hot path.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


def merge_lists(left: list, right: list) -> list:
    """Merge two lists, deduplicating by content if possible."""
    seen = set()
    result = []
    for item in left + right:
        # Simple dedup by string representation
        key = str(item) if not hasattr(item, 'id') else item.id
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def merge_dicts(left: dict, right: dict) -> dict:
    """Merge dicts, with right taking precedence."""
    return {**left, **right}


class PlutusState(BaseModel):
    """Shared state across all agents in the workflow.
    
    Context Engineering Principles:
    - Only summarized/processed data in state
    - Raw data is processed before adding
    - Large datasets go to Qdrant, not state
    """
    
    # Task context
    task_type: str = Field("", description="Current task type")
    task_started_at: datetime = Field(default_factory=datetime.now)
    
    # Portfolio (summarized)
    portfolio_summary: str = Field("", description="Concise portfolio summary")
    portfolio_tickers: list[str] = Field(default_factory=list)
    
    # News (summarized, limited)
    news_digest: str = Field("", description="Summarized news digest")
    top_news_items: Annotated[list[dict[str, Any]], merge_lists] = Field(
        default_factory=list,
        description="Top 5 most relevant news items (summarized)",
    )
    
    # Market data (essential only)
    market_snapshot: Annotated[dict[str, Any], merge_dicts] = Field(
        default_factory=dict,
        description="Key price data for portfolio tickers",
    )
    
    # Analysis results
    risk_signals: Annotated[list[dict[str, Any]], merge_lists] = Field(
        default_factory=list,
        description="Risk signals from analysis",
    )
    opportunities: Annotated[list[dict[str, Any]], merge_lists] = Field(
        default_factory=list,
        description="Investment opportunities found",
    )
    trend_insights: Annotated[list[dict[str, Any]], merge_lists] = Field(
        default_factory=list,
        description="Macro trend insights",
    )
    
    # Agent coordination
    agents_invoked: list[str] = Field(default_factory=list)
    current_agent: str = Field("")
    
    # Errors and warnings
    errors: Annotated[list[str], merge_lists] = Field(default_factory=list)
    warnings: Annotated[list[str], merge_lists] = Field(default_factory=list)
    
    # Final output
    final_report: str = Field("", description="Consolidated report")
    should_notify: bool = Field(False, description="Whether to send notification")
    
    # Messages for agent communication (LangGraph pattern)
    messages: Annotated[list[Any], add_messages] = Field(default_factory=list)
    
    def get_context_for_agent(self, agent_name: str) -> dict[str, Any]:
        """Get relevant context subset for specific agent.
        
        Each agent only gets what it needs - context engineering.
        """
        base_context = {
            "task_type": self.task_type,
            "portfolio_summary": self.portfolio_summary,
            "portfolio_tickers": self.portfolio_tickers,
        }
        
        if agent_name == "portfolio_tracker":
            return {
                **base_context,
                "market_snapshot": self.market_snapshot,
            }
        elif agent_name == "news_monitor":
            return {
                **base_context,
            }
        elif agent_name == "risk_assessor":
            return {
                **base_context,
                "news_digest": self.news_digest,
                "market_snapshot": self.market_snapshot,
            }
        elif agent_name == "opportunity_scout":
            return {
                **base_context,
                "news_digest": self.news_digest,
                "trend_insights": self.trend_insights,
            }
        elif agent_name == "trend_interpreter":
            return {
                **base_context,
                "news_digest": self.news_digest,
            }
        elif agent_name == "market_analyst":
            return {
                **base_context,
                "market_snapshot": self.market_snapshot,
            }
        else:
            # Coordinator or unknown - gets everything
            return self.model_dump(exclude={"messages"})
    
    def add_error(self, agent: str, error: str) -> None:
        """Record an error from an agent."""
        self.errors.append(f"[{agent}] {error}")
    
    def add_warning(self, agent: str, warning: str) -> None:
        """Record a warning from an agent."""
        self.warnings.append(f"[{agent}] {warning}")
