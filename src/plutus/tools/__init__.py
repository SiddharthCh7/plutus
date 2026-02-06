"""Tools package - MCP-portable implementations."""

from plutus.tools.news import fetch_news_for_tickers, NewsProvider
from plutus.tools.market_data import get_stock_prices, MarketDataProvider

__all__ = [
    "fetch_news_for_tickers",
    "NewsProvider",
    "get_stock_prices",
    "MarketDataProvider",
]
