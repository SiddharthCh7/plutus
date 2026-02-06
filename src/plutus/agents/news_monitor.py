"""News Monitor Agent - Fetches and analyzes relevant news."""

from __future__ import annotations

from typing import Any

from plutus.agents.base import BaseAgent
from plutus.logging import get_logger
from plutus.memory import NewsSummarizer
from plutus.tools.news import fetch_news_for_tickers

logger = get_logger(__name__)


class NewsMonitorAgent(BaseAgent):
    """Monitors news for portfolio and market insights.
    
    Responsibilities:
    - Fetch news for portfolio tickers
    - Summarize and filter relevant news
    - Identify market-moving events
    - Store in Qdrant for future retrieval
    """
    
    name = "news_monitor"
    description = "Fetches and analyzes relevant financial news"
    token_budget = 3000
    
    def __init__(self) -> None:
        super().__init__()
        self._summarizer = NewsSummarizer()
    
    def get_system_prompt(self) -> str:
        return """You are a financial news analyst. Your task is to analyze news and extract actionable insights.

For each piece of news, determine:
1. Relevance to the portfolio
2. Sentiment (positive/negative/neutral)
3. Potential impact on stock prices
4. Any immediate action required

Provide a concise digest focusing on the most important news.
Never include full article text - only summarized insights."""

    async def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Fetch and analyze news for portfolio tickers."""
        tickers = context.get("portfolio_tickers", [])
        
        # Fetch news
        logger.info("Fetching news", tickers=tickers)
        raw_news = await fetch_news_for_tickers(tickers)
        
        if not raw_news:
            logger.info("No news found")
            return {
                "news_digest": "No relevant news found.",
                "top_news_items": [],
            }
        
        # Summarize each news item (lossless)
        summarized_items = []
        for item in raw_news[:10]:  # Limit to top 10
            try:
                summary_result = await self._summarizer.summarize(
                    content=item.get("content", item.get("title", "")),
                    context={"portfolio_tickers": tickers},
                )
                
                summarized_items.append({
                    "title": item.get("title"),
                    "source": item.get("source"),
                    "summary": summary_result.summary,
                    "sentiment": summary_result.sentiment,
                    "key_signals": summary_result.key_signals,
                    "entities": summary_result.entities,
                })
            except Exception as e:
                logger.warning("Failed to summarize news", error=str(e))
                continue
        
        # Generate digest
        digest = await self._generate_digest(summarized_items, tickers)
        
        # Store in Qdrant for future retrieval
        # (Done asynchronously, not blocking)
        # await self._store_in_memory(summarized_items)
        
        logger.info(
            "News analyzed",
            total_items=len(raw_news),
            summarized=len(summarized_items),
        )
        
        return {
            "news_digest": digest,
            "top_news_items": summarized_items[:5],
        }
    
    async def _generate_digest(
        self,
        items: list[dict[str, Any]],
        tickers: list[str],
    ) -> str:
        """Generate a concise news digest."""
        if not items:
            return "No significant news."
        
        # Group by sentiment
        positive = [i for i in items if i.get("sentiment") == "positive"]
        negative = [i for i in items if i.get("sentiment") == "negative"]
        neutral = [i for i in items if i.get("sentiment") not in ["positive", "negative"]]
        
        digest_parts = []
        
        if negative:
            digest_parts.append(
                f"⚠️ {len(negative)} negative: " +
                ", ".join(i["summary"][:50] for i in negative[:2])
            )
        
        if positive:
            digest_parts.append(
                f"✅ {len(positive)} positive: " +
                ", ".join(i["summary"][:50] for i in positive[:2])
            )
        
        if neutral:
            digest_parts.append(f"📰 {len(neutral)} neutral items")
        
        return " | ".join(digest_parts)
