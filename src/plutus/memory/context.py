"""Context Builder - Intelligent context engineering with token budgeting.

Core principle: Never dump raw data into LLM context.
Only pass curated, relevant, summarized information.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from plutus.logging import get_logger
from plutus.memory.qdrant import QdrantMemory, get_memory
from plutus.models import Portfolio, NewsItem

logger = get_logger(__name__)

# Token estimates (approximate)
CHARS_PER_TOKEN = 4
DEFAULT_TOKEN_BUDGET = 4000  # Conservative default


@dataclass
class ContextSection:
    """A section of context with its priority."""
    
    name: str
    content: str
    priority: int = 1  # Higher = more important
    tokens: int = 0
    
    def __post_init__(self):
        self.tokens = len(self.content) // CHARS_PER_TOKEN


@dataclass
class BuiltContext:
    """Final built context ready for LLM."""
    
    sections: list[ContextSection] = field(default_factory=list)
    total_tokens: int = 0
    truncated: bool = False
    
    def to_string(self) -> str:
        """Combine all sections into final context string."""
        parts = []
        for section in self.sections:
            parts.append(f"## {section.name}\n{section.content}")
        return "\n\n".join(parts)
    
    def to_messages_format(self) -> str:
        """Format for inclusion in messages."""
        return f"<context>\n{self.to_string()}\n</context>"


class ContextBuilder:
    """Builds optimized context for LLM calls.
    
    Features:
    - Token budgeting per agent
    - Priority-based section inclusion
    - Semantic retrieval from Qdrant
    - Automatic truncation with logging
    
    Usage:
        builder = ContextBuilder(token_budget=4000)
        builder.add_portfolio(portfolio)
        builder.add_news_digest(news_digest)
        await builder.add_relevant_history(query, tickers)
        context = builder.build()
    """
    
    def __init__(
        self,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
        memory: QdrantMemory | None = None,
    ) -> None:
        self._budget = token_budget
        self._memory = memory or get_memory()
        self._sections: list[ContextSection] = []
        self._used_tokens = 0
    
    def add_section(
        self,
        name: str,
        content: str,
        priority: int = 1,
    ) -> "ContextBuilder":
        """Add a context section with priority.
        
        Higher priority sections are kept when truncating.
        """
        section = ContextSection(name=name, content=content, priority=priority)
        self._sections.append(section)
        self._used_tokens += section.tokens
        
        logger.debug(
            "Added context section",
            name=name,
            tokens=section.tokens,
            priority=priority,
        )
        
        return self
    
    def add_portfolio(
        self,
        portfolio: Portfolio,
        priority: int = 3,
    ) -> "ContextBuilder":
        """Add portfolio summary (high priority)."""
        return self.add_section(
            name="Portfolio",
            content=portfolio.to_summary(),
            priority=priority,
        )
    
    def add_news_items(
        self,
        items: list[NewsItem],
        max_items: int = 5,
        priority: int = 2,
    ) -> "ContextBuilder":
        """Add news items (summarized, limited)."""
        # Sort by relevance and take top items
        sorted_items = sorted(
            items,
            key=lambda x: x.relevance_score,
            reverse=True,
        )[:max_items]
        
        content = "\n".join(
            item.to_context_string() for item in sorted_items
        )
        
        return self.add_section(
            name="Recent News",
            content=content,
            priority=priority,
        )
    
    def add_raw_text(
        self,
        name: str,
        text: str,
        priority: int = 1,
    ) -> "ContextBuilder":
        """Add raw text section."""
        return self.add_section(name=name, content=text, priority=priority)
    
    async def add_relevant_history(
        self,
        query_embedding: list[float],
        tickers: list[str] | None = None,
        category: str | None = None,
        max_items: int = 3,
        priority: int = 1,
    ) -> "ContextBuilder":
        """Add relevant items from vector memory.
        
        Uses semantic search to find relevant past context.
        """
        results = await self._memory.search(
            query_embedding=query_embedding,
            tickers=tickers,
            category=category,
            limit=max_items,
        )
        
        if results:
            content = "\n---\n".join(
                r.get("content", "") for r in results
            )
            self.add_section(
                name="Relevant History",
                content=content,
                priority=priority,
            )
        
        return self
    
    def build(self) -> BuiltContext:
        """Build the final context respecting token budget.
        
        If over budget:
        1. Sort by priority (highest first)
        2. Include sections until budget exhausted
        3. Truncate last section if needed
        """
        if self._used_tokens <= self._budget:
            # Under budget, include everything
            return BuiltContext(
                sections=self._sections,
                total_tokens=self._used_tokens,
                truncated=False,
            )
        
        # Over budget, need to prioritize
        logger.warning(
            "Context over budget, truncating",
            used=self._used_tokens,
            budget=self._budget,
        )
        
        # Sort by priority (highest first)
        sorted_sections = sorted(
            self._sections,
            key=lambda x: x.priority,
            reverse=True,
        )
        
        included: list[ContextSection] = []
        remaining_budget = self._budget
        
        for section in sorted_sections:
            if section.tokens <= remaining_budget:
                included.append(section)
                remaining_budget -= section.tokens
            elif remaining_budget > 100:  # Truncate if meaningful space
                # Truncate section
                char_limit = remaining_budget * CHARS_PER_TOKEN
                truncated_content = section.content[:char_limit] + "..."
                truncated_section = ContextSection(
                    name=section.name,
                    content=truncated_content,
                    priority=section.priority,
                )
                included.append(truncated_section)
                remaining_budget = 0
                break
        
        total = sum(s.tokens for s in included)
        
        logger.info(
            "Context built with truncation",
            included_sections=len(included),
            total_tokens=total,
            dropped_sections=len(self._sections) - len(included),
        )
        
        return BuiltContext(
            sections=included,
            total_tokens=total,
            truncated=True,
        )
    
    def reset(self) -> "ContextBuilder":
        """Reset builder for reuse."""
        self._sections = []
        self._used_tokens = 0
        return self
