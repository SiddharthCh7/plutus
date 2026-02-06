"""Market data tools - MCP-portable design.

Current: Yahoo Finance (yfinance)
Future: Can be replaced with MCP server or other providers
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import yfinance as yf

from plutus.logging import get_logger

logger = get_logger(__name__)


class MarketDataProvider(ABC):
    """Abstract market data provider - MCP-ready interface."""
    
    @abstractmethod
    async def get_prices(
        self,
        tickers: list[str],
    ) -> dict[str, dict[str, Any]]:
        """Get current prices for tickers.
        
        Returns:
            Dict mapping ticker to price data:
            {
                "AAPL": {
                    "price": 150.25,
                    "change_percent": 1.5,
                    "volume": 50000000,
                    "market_cap": 2500000000000,
                }
            }
        """
        pass


class YahooFinanceMarket(MarketDataProvider):
    """Yahoo Finance market data provider."""
    
    async def get_prices(
        self,
        tickers: list[str],
    ) -> dict[str, dict[str, Any]]:
        """Get current prices from Yahoo Finance."""
        result = {}
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info or {}
                
                # Get current price
                price = info.get("currentPrice") or info.get("regularMarketPrice", 0)
                previous_close = info.get("previousClose", price)
                
                if previous_close and previous_close > 0:
                    change_percent = ((price - previous_close) / previous_close) * 100
                else:
                    change_percent = 0
                
                result[ticker] = {
                    "price": price,
                    "change_percent": round(change_percent, 2),
                    "volume": info.get("volume", 0),
                    "market_cap": info.get("marketCap", 0),
                    "pe_ratio": info.get("forwardPE") or info.get("trailingPE"),
                    "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
                    "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
                }
                
            except Exception as e:
                logger.warning(
                    "Failed to fetch price",
                    ticker=ticker,
                    error=str(e),
                )
                result[ticker] = {
                    "price": None,
                    "error": str(e),
                }
        
        logger.info("Fetched prices", tickers=len(result))
        
        return result


# Default provider instance
_provider: MarketDataProvider | None = None


def get_market_provider() -> MarketDataProvider:
    """Get the configured market data provider."""
    global _provider
    if _provider is None:
        _provider = YahooFinanceMarket()
    return _provider


async def get_stock_prices(
    tickers: list[str],
) -> dict[str, dict[str, Any]]:
    """Get stock prices using configured provider.
    
    This is the main entry point for price data.
    """
    provider = get_market_provider()
    return await provider.get_prices(tickers)
