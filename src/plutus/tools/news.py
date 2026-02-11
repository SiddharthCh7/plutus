"""News fetching tools - MCP-portable design.

Current: Yahoo Finance (yfinance)
Future: Can be replaced with MCP server or other providers
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any

import yfinance as yf

from plutus.logging import get_logger

logger = get_logger(__name__)


class NewsProvider(ABC):
    """Abstract news provider - MCP-ready interface.
    
    Implement this interface for new providers:
    - MCPNewsProvider (future)
    - NewsAPIProvider
    - RSSFeedProvider
    """
    
    @abstractmethod
    async def fetch(
        self,
        tickers: list[str],
        days_back: int = 7,
    ) -> list[dict[str, Any]]:
        """Fetch news for given tickers.
        
        Args:
            tickers: Stock ticker symbols
            days_back: How many days of news to fetch
            
        Returns:
            List of news items with title, content, source, url, published_at
        """
        pass


class YahooFinanceNews(NewsProvider):
    """Yahoo Finance news provider using yfinance."""
    
    async def fetch(
        self,
        tickers: list[str],
        days_back: int = 7,
    ) -> list[dict[str, Any]]:
        """Fetch news from Yahoo Finance."""
        all_news = []
        total_fetched = 0
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                news = stock.news or []
                total_fetched += len(news)
                
                for item in news:
                    # yfinance nests data under 'content' key
                    content = item.get("content", item)
                    
                    # Parse timestamp (ISO format string, e.g. "2026-02-10T14:46:24Z")
                    pub_date_str = content.get("pubDate") or content.get("displayTime", "")
                    if pub_date_str:
                        try:
                            pub_date = datetime.fromisoformat(
                                pub_date_str.replace("Z", "+00:00")
                            ).replace(tzinfo=None)
                        except (ValueError, TypeError):
                            pub_date = datetime.now()
                    else:
                        # Fallback for legacy format
                        pub_time = content.get("providerPublishTime", 0)
                        pub_date = datetime.fromtimestamp(pub_time) if pub_time else datetime.now()
                    
                    if pub_date < cutoff_date:
                        continue
                    
                    all_news.append({
                        "title": content.get("title", ""),
                        "content": content.get("summary", content.get("title", "")),
                        "published_at": pub_date.isoformat(),
                        "related_tickers": [ticker],
                    })
                    
            except Exception as e:
                logger.warning(
                    "Failed to fetch news",
                    ticker=ticker,
                    error=str(e),
                )
                continue
        
        # Sort by date, newest first
        all_news.sort(
            key=lambda x: x.get("published_at", ""),
            reverse=True,
        )
        
        logger.info(
            "Fetched news",
            tickers=len(tickers),
            items_found=len(all_news),
            items_dropped=total_fetched - len(all_news),
        )
        
        return all_news


# Default provider instance
_provider: NewsProvider | None = None


def get_news_provider() -> NewsProvider:
    """Get the configured news provider."""
    global _provider
    if _provider is None:
        _provider = YahooFinanceNews()
    return _provider


async def fetch_news_for_tickers(
    tickers: list[str],
    days_back: int = 7,
) -> list[dict[str, Any]]:
    """Fetch news for given tickers using configured provider.
    
    This is the main entry point for news fetching.
    """
    provider = get_news_provider()
    return await provider.fetch(tickers, days_back)
