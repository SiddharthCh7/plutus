"""Base agent class with common functionality."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from plutus.llm import get_model_router
from plutus.logging import get_logger
from plutus.memory import ContextBuilder, get_memory

logger = get_logger(__name__)


class BaseAgent(ABC):
    """Base class for all Plutus agents.
    
    Provides:
    - LLM access via model router
    - Context building with token budgets
    - Structured logging
    - Error handling
    """
    
    # Override in subclasses
    name: str = "base"
    description: str = "Base agent"
    token_budget: int = 4000
    
    def __init__(self) -> None:
        self._router = get_model_router()
        self._memory = get_memory()
    
    @property
    def model(self) -> BaseChatModel:
        """Get the LLM for this agent."""
        return self._router.get_model(agent_name=self.name)
    
    @abstractmethod
    async def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Execute the agent's task.
        
        Args:
            context: Agent-specific context from state
            
        Returns:
            Dict of updates to apply to state
        """
        pass
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        pass
    
    async def invoke_llm(
        self,
        user_message: str,
        context_str: str = "",
    ) -> str:
        """Invoke LLM with system prompt and context.
        
        Args:
            user_message: The user/task message
            context_str: Pre-built context string
            
        Returns:
            LLM response text
        """
        system_prompt = self.get_system_prompt()
        
        # Inject context into system prompt if provided
        if context_str:
            system_prompt = f"{system_prompt}\n\n{context_str}"
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]
        
        logger.debug(
            "Invoking LLM",
            agent=self.name,
            prompt_length=len(system_prompt),
            message_length=len(user_message),
        )
        
        response = await self.model.ainvoke(messages)
        
        logger.debug(
            "LLM response received",
            agent=self.name,
            response_length=len(response.content),
        )
        
        return response.content
    
    def create_context_builder(self) -> ContextBuilder:
        """Create a context builder with this agent's token budget."""
        return ContextBuilder(
            token_budget=self.token_budget,
            memory=self._memory,
        )
