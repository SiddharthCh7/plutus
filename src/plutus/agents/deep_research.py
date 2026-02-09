"""Deep Research Agent - In-depth analysis of a single ticker.

Performs comprehensive research including:
- Fundamental analysis (P/E, EPS, book value, etc.)
- Growth metrics (revenue, profit growth)
- Sector analysis
- Valuation assessment (undervalued/overvalued)
- Red flags identification
- Investment summary
"""

from __future__ import annotations

from typing import Any

from plutus.agents.base import BaseAgent
from plutus.logging import get_logger
from plutus.tools.market_data import get_stock_prices

logger = get_logger(__name__)


class DeepResearchAgent(BaseAgent):
    """Deep research agent for comprehensive ticker analysis.
    
    Responsibilities:
    - Fetch detailed fundamentals
    - Analyze growth trajectory
    - Compare to sector peers
    - Identify valuation status
    - Flag potential risks
    - Provide investment summary
    """
    
    name = "deep_research"
    description = "Performs in-depth research and analysis on a single ticker"
    token_budget = 6000  # Larger budget for comprehensive analysis
    
    def get_system_prompt(self) -> str:
        return """You are a senior equity research analyst. Your task is to provide comprehensive analysis of a stock.

Format your analysis as follows:

## 📊 Company Overview
Brief description of what the company does, its market position, and key products/services.

## 💰 Key Financial Metrics
Analyze the provided metrics and explain what they mean for investors:
- P/E Ratio: Is it reasonable for the sector?
- EPS: Positive or negative? Growing?
- Book Value: Asset backing per share
- Debt levels: Concerning or manageable?

## 📈 Growth Analysis
- Revenue growth trajectory
- Profit margin trends
- Market share changes
- Expansion plans

## 🏭 Sector Position
- Industry outlook
- Competitive advantages (or lack thereof)
- Key competitors
- Market tailwinds/headwinds

## 💎 Valuation Assessment
Based on fundamentals, determine if the stock is:
- UNDERVALUED: Trading below intrinsic value, potential upside
- FAIRLY VALUED: Price reflects fundamentals
- OVERVALUED: Trading above intrinsic value, limited upside

## 🚩 Red Flags
List any concerns investors should watch:
- Declining revenues
- High debt
- Management issues
- Sector headwinds
- Regulatory risks

## ✅ Investment Summary
Provide a clear recommendation with reasoning:
- TIME HORIZON: Short/Medium/Long term
- RISK LEVEL: Low/Medium/High
- VERDICT: BUY / HOLD / AVOID
- KEY CATALYST: What could drive the stock up
- KEY RISK: What could drive it down

Be specific, data-driven, and balanced. Always mention both upside potential and risks."""

    async def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Perform deep research on a ticker."""
        ticker = context.get("ticker")
        
        if not ticker:
            logger.warning("No ticker provided for deep research")
            return {"research_report": "No ticker provided for research."}
        
        logger.info("Performing deep research", ticker=ticker)
        
        # Fetch comprehensive data
        market_data = await self._fetch_detailed_data(ticker)
        
        if not market_data or market_data.get("error"):
            return {
                "research_report": f"Unable to fetch data for {ticker}. Error: {market_data.get('error', 'Unknown')}"
            }
        
        # Build context with all available data
        builder = self.create_context_builder()
        builder.add_raw_text(
            "Stock Data",
            self._format_market_data(ticker, market_data),
            priority=3,
        )
        built_context = builder.build()
        
        # Invoke LLM for comprehensive analysis
        user_message = f"""Perform comprehensive research and analysis on: {ticker}

Provide a detailed investment research report covering all aspects:
fundamentals, growth, sector analysis, valuation, red flags, and investment verdict.

Be specific and data-driven using the provided financial data."""

        response = await self.invoke_llm(
            user_message=user_message,
            context_str=built_context.to_messages_format(),
        )
        
        logger.info("Deep research completed", ticker=ticker, length=len(response))
        
        return {
            "research_report": response,
            "ticker": ticker,
            "market_data": market_data,
        }
    
    async def _fetch_detailed_data(self, ticker: str) -> dict[str, Any]:
        """Fetch detailed market data for a ticker."""
        import yfinance as yf
        
        try:
            stock = yf.Ticker(ticker)
            info = stock.info or {}
            
            return {
                "price": info.get("currentPrice") or info.get("regularMarketPrice"),
                "previous_close": info.get("previousClose"),
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "eps": info.get("trailingEps"),
                "book_value": info.get("bookValue"),
                "price_to_book": info.get("priceToBook"),
                "dividend_yield": info.get("dividendYield"),
                "revenue": info.get("totalRevenue"),
                "gross_profit": info.get("grossProfits"),
                "operating_margin": info.get("operatingMargins"),
                "profit_margin": info.get("profitMargins"),
                "debt_to_equity": info.get("debtToEquity"),
                "current_ratio": info.get("currentRatio"),
                "return_on_equity": info.get("returnOnEquity"),
                "return_on_assets": info.get("returnOnAssets"),
                "52_week_high": info.get("fiftyTwoWeekHigh"),
                "52_week_low": info.get("fiftyTwoWeekLow"),
                "50_day_avg": info.get("fiftyDayAverage"),
                "200_day_avg": info.get("twoHundredDayAverage"),
                "volume": info.get("volume"),
                "avg_volume": info.get("averageVolume"),
                "beta": info.get("beta"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "company_name": info.get("longName") or info.get("shortName"),
                "description": info.get("longBusinessSummary", "")[:500],
                "recommendation": info.get("recommendationKey"),
                "target_price": info.get("targetMeanPrice"),
            }
        except Exception as e:
            logger.error("Failed to fetch detailed data", ticker=ticker, error=str(e))
            return {"error": str(e)}
    
    def _format_market_data(self, ticker: str, data: dict[str, Any]) -> str:
        """Format market data for LLM context."""
        lines = [f"Stock: {ticker} ({data.get('company_name', 'Unknown')})"]
        
        if data.get("description"):
            lines.append(f"Business: {data['description']}")
        
        lines.append(f"\nSector: {data.get('sector', 'N/A')}")
        lines.append(f"Industry: {data.get('industry', 'N/A')}")
        
        lines.append("\n--- PRICE DATA ---")
        lines.append(f"Current Price: {data.get('price', 'N/A')}")
        lines.append(f"52-Week High: {data.get('52_week_high', 'N/A')}")
        lines.append(f"52-Week Low: {data.get('52_week_low', 'N/A')}")
        lines.append(f"50-Day Avg: {data.get('50_day_avg', 'N/A')}")
        lines.append(f"200-Day Avg: {data.get('200_day_avg', 'N/A')}")
        
        lines.append("\n--- VALUATION ---")
        lines.append(f"Market Cap: {data.get('market_cap', 'N/A')}")
        lines.append(f"P/E Ratio (TTM): {data.get('pe_ratio', 'N/A')}")
        lines.append(f"Forward P/E: {data.get('forward_pe', 'N/A')}")
        lines.append(f"EPS: {data.get('eps', 'N/A')}")
        lines.append(f"Book Value: {data.get('book_value', 'N/A')}")
        lines.append(f"Price to Book: {data.get('price_to_book', 'N/A')}")
        
        lines.append("\n--- FINANCIALS ---")
        lines.append(f"Revenue: {data.get('revenue', 'N/A')}")
        lines.append(f"Gross Profit: {data.get('gross_profit', 'N/A')}")
        lines.append(f"Operating Margin: {self._format_percent(data.get('operating_margin'))}")
        lines.append(f"Profit Margin: {self._format_percent(data.get('profit_margin'))}")
        
        lines.append("\n--- BALANCE SHEET ---")
        lines.append(f"Debt to Equity: {data.get('debt_to_equity', 'N/A')}")
        lines.append(f"Current Ratio: {data.get('current_ratio', 'N/A')}")
        lines.append(f"ROE: {self._format_percent(data.get('return_on_equity'))}")
        lines.append(f"ROA: {self._format_percent(data.get('return_on_assets'))}")
        
        lines.append("\n--- OTHER ---")
        lines.append(f"Dividend Yield: {self._format_percent(data.get('dividend_yield'))}")
        lines.append(f"Beta: {data.get('beta', 'N/A')}")
        lines.append(f"Analyst Target: {data.get('target_price', 'N/A')}")
        lines.append(f"Recommendation: {data.get('recommendation', 'N/A')}")
        
        return "\n".join(lines)
    
    def _format_percent(self, value: float | None) -> str:
        """Format a decimal as percentage."""
        if value is None:
            return "N/A"
        return f"{value * 100:.2f}%"
