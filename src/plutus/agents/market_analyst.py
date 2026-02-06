"""Market Analyst Agent - Technical and fundamental analysis."""

from __future__ import annotations

from typing import Any

from plutus.agents.base import BaseAgent
from plutus.logging import get_logger

logger = get_logger(__name__)


class MarketAnalystAgent(BaseAgent):
    """Analyzes market data for technical and fundamental insights.
    
    Responsibilities:
    - Technical analysis (trends, momentum)
    - Valuation metrics
    - Sector comparisons
    """
    
    name = "market_analyst"
    description = "Performs technical and fundamental market analysis"
    token_budget = 3000
    
    def get_system_prompt(self) -> str:
        return """You are a market analyst specializing in technical and fundamental analysis.

Given market data, analyze:
1. Price trends (bullish/bearish/sideways)
2. Key support/resistance levels
3. Valuation relative to sector
4. Volume patterns

Be concise. Focus on actionable insights for long-term investors."""

    async def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Analyze market data for portfolio tickers."""
        market_snapshot = context.get("market_snapshot", {})
        tickers = context.get("portfolio_tickers", [])
        
        if not market_snapshot:
            logger.warning("No market data to analyze")
            return {"analysis": {}}
        
        # Build analysis for each ticker
        analysis = {}
        for ticker in tickers:
            ticker_data = market_snapshot.get(ticker, {})
            if ticker_data:
                change = ticker_data.get("change_percent", 0)
                
                # Simple trend classification
                if change > 2:
                    trend = "bullish"
                elif change < -2:
                    trend = "bearish"
                else:
                    trend = "sideways"
                
                analysis[ticker] = {
                    "trend": trend,
                    "change_percent": change,
                    "analysis_note": f"{ticker} showing {trend} movement",
                }
        
        logger.info("Market analyzed", tickers=len(analysis))
        
        return {"analysis": analysis}
