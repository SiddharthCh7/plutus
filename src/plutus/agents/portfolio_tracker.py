"""Portfolio Tracker Agent - Monitors holdings and calculates P&L."""

from __future__ import annotations

from typing import Any

from plutus.agents.base import BaseAgent
from plutus.logging import get_logger
from plutus.tools.market_data import get_stock_prices

logger = get_logger(__name__)


class PortfolioTrackerAgent(BaseAgent):
    """Tracks portfolio holdings and performance.
    
    Responsibilities:
    - Fetch current prices for holdings
    - Calculate P&L for each position
    - Identify significant changes
    """
    
    name = "portfolio_tracker"
    description = "Monitors portfolio holdings and calculates performance"
    token_budget = 2000  # Lightweight agent
    
    def get_system_prompt(self) -> str:
        return """You are a portfolio tracking assistant. Your task is to analyze portfolio performance.

Given the portfolio holdings and current market prices, provide:
1. Current value and P&L for each holding
2. Notable changes (>5% movement)
3. Any holdings approaching target prices

Be concise and factual. Focus on actionable information."""

    async def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Fetch prices and calculate portfolio metrics."""
        tickers = context.get("portfolio_tickers", [])
        
        if not tickers:
            logger.warning("No tickers to track")
            return {"market_snapshot": {}}
        
        # Fetch current prices
        logger.info("Fetching prices", tickers=tickers)
        prices = await get_stock_prices(tickers)
        
        # Build market snapshot
        market_snapshot = {}
        for ticker, price_data in prices.items():
            market_snapshot[ticker] = {
                "current_price": price_data.get("price"),
                "change_percent": price_data.get("change_percent"),
                "volume": price_data.get("volume"),
            }
        
        logger.info("Portfolio tracked", tickers_fetched=len(prices))
        
        return {
            "market_snapshot": market_snapshot,
        }
