"""Opportunity Scout Agent - Finds undervalued investment opportunities."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from plutus.agents.base import BaseAgent
from plutus.logging import get_logger

logger = get_logger(__name__)

# Load info.json for sectors to scout
INFO_PATH = Path(__file__).parent.parent.parent.parent / "data" / "info.json"
with open(INFO_PATH) as f:
    INFO_DATA = json.load(f)


class OpportunityScoutAgent(BaseAgent):
    """Scouts for investment opportunities.
    
    Responsibilities:
    - Find undervalued stocks
    - Identify beneficiaries of trends
    - Evaluate long-term potential
    
    Focus areas (per user):
    - Green energy
    - Semiconductors
    - Commodities
    """
    
    name = "opportunity_scout"
    description = "Discovers undervalued stocks with long-term potential"
    token_budget = 4000
    
    def get_system_prompt(self) -> str:
        sectors = INFO_DATA.get("sectors_to_scout", [])
        sectors_list = "\n".join(f"- {sector}" for sector in sectors)
        return f"""You are an investment opportunity scout focused on LONG-TERM value investing.

Your task is to identify undervalued stocks with high future potential.

Focus areas:
{sectors_list}

For each opportunity, provide:
- Ticker and company name
- Investment thesis (why it's undervalued)
- Catalysts (what will drive growth)
- Risks
- Time horizon

Output format:
TICKER|COMPANY|SECTOR|SCORE|THESIS

Example:
TSMC|Taiwan Semiconductor|Semiconductors|0.85|Primary beneficiary of AI chip demand, undervalued relative to growth

Be selective - only recommend high-conviction opportunities."""

    async def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Scout for investment opportunities."""
        news_digest = context.get("news_digest", "")
        trend_insights = context.get("trend_insights", [])
        
        # Build context with trends
        builder = self.create_context_builder()
        builder.add_raw_text("News", news_digest, priority=2)
        
        if trend_insights:
            trends_text = "\n".join(
                f"- {t.get('trend_name')}: {t.get('reasoning', '')[:100]}"
                for t in trend_insights
            )
            builder.add_raw_text("Trends", trends_text, priority=3)
        
        built_context = builder.build()
        
        # Invoke LLM
        user_message = "Find investment opportunities based on the provided news and trends."

        response = await self.invoke_llm(
            user_message=user_message,
            context_str=built_context.to_messages_format(),
        )
        
        # Parse opportunities
        opportunities = self._parse_opportunities(response)
        
        logger.info("Opportunities found", count=len(opportunities))
        
        return {"opportunities": opportunities}
    
    def _parse_opportunities(self, response: str) -> list[dict[str, Any]]:
        """Parse LLM response into structured opportunities."""
        opportunities = []
        
        for line in response.strip().split("\n"):
            line = line.strip()
            if "|" in line:
                parts = line.split("|")
                if len(parts) >= 5:
                    try:
                        score = float(parts[3].strip())
                    except ValueError:
                        score = 0.5
                    
                    opportunities.append({
                        "ticker": parts[0].strip().upper(),
                        "company_name": parts[1].strip(),
                        "sector": parts[2].strip(),
                        "opportunity_score": score,
                        "thesis": parts[4].strip(),
                    })
        
        return opportunities
