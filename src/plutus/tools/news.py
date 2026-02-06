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
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                news = stock.news or []
                
                for item in news:
                    # Parse timestamp
                    pub_time = item.get("providerPublishTime", 0)
                    pub_date = datetime.fromtimestamp(pub_time)
                    
                    if pub_date < cutoff_date:
                        continue
                    
                    all_news.append({
                        "title": item.get("title", ""),
                        "content": item.get("summary", item.get("title", "")),
                        "source": item.get("publisher", "Yahoo Finance"),
                        "url": item.get("link", ""),
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
            items=len(all_news),
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
