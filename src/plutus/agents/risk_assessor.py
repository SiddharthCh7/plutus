"""Risk Assessor Agent - Generates exit signals based on analysis."""

from __future__ import annotations

from typing import Any

from datetime import datetime

from plutus.agents.base import BaseAgent
from plutus.logging import get_logger

logger = get_logger(__name__)


class RiskAssessorAgent(BaseAgent):
    """Assesses risk and generates exit signals.
    
    Responsibilities:
    - Analyze news sentiment for holdings
    - Evaluate price movements
    - Generate actionable signals (EXIT, REDUCE, HOLD, ACCUMULATE)
    """
    
    name = "risk_assessor"
    description = "Evaluates portfolio risk and generates exit signals"
    token_budget = 4000
    
    def get_system_prompt(self) -> str:
        return """You are a risk assessment specialist for long-term investors.

Given the portfolio holdings, news digest, and market data, assess the risk for each holding.

For each holding, provide a signal:
- EXIT: Strong sell signal, immediate action recommended
- REDUCE: Consider reducing position
- HOLD: Maintain current position
- ACCUMULATE: Consider buying more

Consider:
1. Negative news patterns
2. Price decline momentum
3. Sector-wide concerns
4. Fundamental deterioration signals

Output format (one per line):
TICKER|SIGNAL|CONFIDENCE|REASON

Example:
AAPL|HOLD|0.8|Strong fundamentals despite market volatility
TSLA|REDUCE|0.6|Increasing competition concerns in EV space"""

    async def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Assess risk for portfolio holdings."""
        portfolio_summary = context.get("portfolio_summary", "")
        news_digest = context.get("news_digest", "")
        market_snapshot = context.get("market_snapshot", {})
        tickers = context.get("portfolio_tickers", [])
        
        if not tickers:
            logger.warning("No tickers to assess")
            return {"risk_signals": []}
        
        # Build context
        builder = self.create_context_builder()
        builder.add_raw_text(
            "Current Date",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            priority=4,
        )
        builder.add_raw_text("Portfolio", portfolio_summary, priority=3)
        builder.add_raw_text("News", news_digest, priority=2)
        builder.add_raw_text(
            "Market",
            str(market_snapshot),
            priority=1,
        )
        built_context = builder.build()
        
        # Invoke LLM for risk assessment
        user_message = f"""Assess risk for these holdings: {', '.join(tickers)}

Provide a risk signal for each ticker."""

        response = await self.invoke_llm(
            user_message=user_message,
            context_str=built_context.to_messages_format(),
        )
        
        # Parse response
        signals = self._parse_signals(response, tickers)
        
        logger.info("Risk assessed", signals_generated=len(signals))
        
        return {"risk_signals": signals}
    
    def _parse_signals(
        self,
        response: str,
        tickers: list[str],
    ) -> list[dict[str, Any]]:
        """Parse LLM response into structured signals."""
        signals = []
        
        for line in response.strip().split("\n"):
            line = line.strip()
            if "|" in line:
                parts = line.split("|")
                if len(parts) >= 4:
                    ticker = parts[0].strip().upper()
                    signal = parts[1].strip().lower()
                    
                    try:
                        confidence = float(parts[2].strip())
                    except ValueError:
                        confidence = 0.5
                    
                    reason = parts[3].strip()
                    
                    if ticker in [t.upper() for t in tickers]:
                        signals.append({
                            "ticker": ticker,
                            "signal": signal,
                            "confidence": confidence,
                            "primary_reason": reason,
                        })
        
        # Ensure all tickers have a signal
        assessed_tickers = {s["ticker"] for s in signals}
        for ticker in tickers:
            if ticker.upper() not in assessed_tickers:
                signals.append({
                    "ticker": ticker.upper(),
                    "signal": "hold",
                    "confidence": 0.5,
                    "primary_reason": "No specific concerns identified",
                })
        
        return signals
