"""Trend Interpreter Agent - The 'Intelligent Investor' logic.

Decodes macro trends and identifies indirect beneficiaries.
Example: AI boom → data centers → cooling systems → power infrastructure
"""

from __future__ import annotations

from typing import Any

from plutus.agents.base import BaseAgent
from plutus.logging import get_logger

logger = get_logger(__name__)


class TrendInterpreterAgent(BaseAgent):
    """Interprets macro trends and finds indirect beneficiaries.
    
    This is the "Intelligent Investor" logic:
    - When everyone invests in X, find Y and Z that benefit from X
    - Example: AI → GPUs → Semiconductors → Energy for data centers
    
    Focus on second and third-order effects.
    """
    
    name = "trend_interpreter"
    description = "Decodes macro trends and identifies indirect beneficiaries"
    token_budget = 4000
    
    def get_system_prompt(self) -> str:
        return """You are a macro trend analyst specializing in identifying INDIRECT beneficiaries of major trends.

Your approach (The Intelligent Investor logic):
1. Identify the obvious trend (e.g., AI boom)
2. Map the direct beneficiaries (NVIDIA, Microsoft)
3. Trace the value chain to find INDIRECT beneficiaries
4. These are often underpriced because the market hasn't connected the dots

Think in terms of:
- Supply chains
- Infrastructure requirements
- Supporting industries
- Geographic beneficiaries

Example analysis:
Trend: AI Infrastructure Spending
Direct: NVIDIA, AMD, Microsoft
Indirect Chain:
  → Semiconductors need fabs → TSMC, ASML
  → Data centers need power → NextEra, utility companies
  → Data centers need cooling → Vertiv, Modine
  → Data centers need land → REITs in specific locations

Output format:
TREND|DESCRIPTION|DIRECT|INDIRECT|REASONING|CONFIDENCE

Where DIRECT and INDIRECT are comma-separated ticker lists."""

    async def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Interpret news for macro trends and indirect beneficiaries."""
        news_digest = context.get("news_digest", "")
        
        if not news_digest:
            logger.warning("No news to interpret")
            return {"trend_insights": []}
        
        # Build context
        builder = self.create_context_builder()
        builder.add_raw_text("Recent News", news_digest, priority=3)
        built_context = builder.build()
        
        # Invoke LLM for trend analysis
        user_message = """Analyze recent news for major macro trends.

For each significant trend:
1. Identify obvious/direct beneficiaries
2. Trace the value chain to find INDIRECT beneficiaries
3. These indirect plays are often the smarter investments

Focus on:
- Trade deals and their beneficiaries
- Infrastructure investments (who builds the infrastructure?)
- Sector rotation opportunities
- Emerging technology adoption chains

Identify 2-3 trends with their indirect beneficiaries."""

        response = await self.invoke_llm(
            user_message=user_message,
            context_str=built_context.to_messages_format(),
        )
        
        # Parse trends
        trends = self._parse_trends(response)
        
        logger.info("Trends interpreted", count=len(trends))
        
        return {"trend_insights": trends}
    
    def _parse_trends(self, response: str) -> list[dict[str, Any]]:
        """Parse LLM response into structured trend insights."""
        trends = []
        
        for line in response.strip().split("\n"):
            line = line.strip()
            if "|" in line and line.count("|") >= 5:
                parts = line.split("|")
                if len(parts) >= 6:
                    try:
                        confidence = float(parts[5].strip())
                    except ValueError:
                        confidence = 0.5
                    
                    direct = [t.strip() for t in parts[2].split(",") if t.strip()]
                    indirect = [t.strip() for t in parts[3].split(",") if t.strip()]
                    
                    trends.append({
                        "trend_name": parts[0].strip(),
                        "description": parts[1].strip(),
                        "direct_beneficiaries": direct,
                        "indirect_beneficiaries": indirect,
                        "reasoning": parts[4].strip(),
                        "confidence": confidence,
                    })
        
        return trends
