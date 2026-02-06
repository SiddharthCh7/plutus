"""Coordinator Agent - Orchestrates the multi-agent workflow."""

from __future__ import annotations

from typing import Any

from plutus.agents.base import BaseAgent
from plutus.logging import get_logger

logger = get_logger(__name__)


class CoordinatorAgent(BaseAgent):
    """Coordinates the multi-agent workflow.
    
    This agent is mostly handled by the LangGraph workflow,
    but provides additional coordination logic if needed.
    """
    
    name = "coordinator"
    description = "Orchestrates specialist agents and aggregates insights"
    token_budget = 2000
    
    def get_system_prompt(self) -> str:
        return """You are the coordinator for Plutus, a personal financial portfolio manager.

Your role is to:
1. Ensure all relevant agents are invoked for the task
2. Aggregate insights from specialist agents
3. Prioritize alerts and opportunities
4. Generate actionable recommendations

Always maintain context efficiency - summarize, don't dump data."""

    async def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Coordinate agent execution.
        
        Note: Most coordination is handled by LangGraph workflow.
        This method is for any additional coordination logic.
        """
        task_type = context.get("task_type", "")
        
        logger.info("Coordinator received context", task_type=task_type)
        
        # Coordination is primarily handled by the workflow
        # This can be extended for more complex coordination
        
        return {}
