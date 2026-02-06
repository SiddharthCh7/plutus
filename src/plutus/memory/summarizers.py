"""Intelligent Summarizers for context engineering.

Key principle: LOSSLESS summarization - never lose important information.
Extracts and preserves critical signals while reducing token count.

Future: Support fine-tuned summarization models from OpenRouter/Ollama.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from plutus.llm import get_model_router
from plutus.logging import get_logger

logger = get_logger(__name__)


class SummaryResult(BaseModel):
    """Result of summarization with metadata."""
    
    summary: str = Field(..., description="Condensed content")
    key_signals: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    sentiment: str | None = Field(None)
    original_length: int = Field(0)
    summary_length: int = Field(0)
    
    @property
    def compression_ratio(self) -> float:
        """How much the content was compressed."""
        if self.original_length == 0:
            return 0
        return 1 - (self.summary_length / self.original_length)


class Summarizer(ABC):
    """Base class for intelligent summarizers.
    
    Design principles:
    1. NEVER lose critical information
    2. Extract and preserve key signals
    3. Maintain entity references
    4. Keep sentiment intact
    """
    
    def __init__(self) -> None:
        self._router = get_model_router()
    
    @abstractmethod
    async def summarize(self, content: str, context: dict[str, Any] | None = None) -> SummaryResult:
        """Summarize content while preserving critical information."""
        pass
    
    def _get_model(self):
        """Get the summarization model."""
        return self._router.get_summarization_model()


class NewsSummarizer(Summarizer):
    """Summarizer specialized for financial news.
    
    Preserves:
    - Company/ticker mentions
    - Numerical data (prices, percentages)
    - Sentiment indicators
    - Action signals (buy/sell/hold implications)
    """
    
    SYSTEM_PROMPT = """You are a financial news summarizer. Your task is to create LOSSLESS summaries that preserve ALL critical information.

RULES:
1. NEVER omit: company names, ticker symbols, numerical values, dates
2. PRESERVE: sentiment (positive/negative/neutral), urgency indicators
3. EXTRACT: key signals that could impact stock prices
4. MAINTAIN: causal relationships (X led to Y)
5. BE CONCISE: remove fluff, keep substance

OUTPUT FORMAT:
Summary: [1-2 sentence summary]
Signals: [comma-separated key signals]
Entities: [company names and tickers mentioned]
Sentiment: [positive/negative/neutral/mixed]"""

    async def summarize(
        self,
        content: str,
        context: dict[str, Any] | None = None,
    ) -> SummaryResult:
        """Summarize news article while preserving critical signals."""
        model = self._get_model()
        
        # Build prompt with optional context
        user_prompt = f"Summarize this financial news:\n\n{content}"
        if context:
            portfolio_tickers = context.get("portfolio_tickers", [])
            if portfolio_tickers:
                user_prompt += f"\n\nFocus on relevance to: {', '.join(portfolio_tickers)}"
        
        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
        
        response = await model.ainvoke(messages)
        parsed = self._parse_response(response.content)
        
        return SummaryResult(
            summary=parsed["summary"],
            key_signals=parsed["signals"],
            entities=parsed["entities"],
            sentiment=parsed["sentiment"],
            original_length=len(content),
            summary_length=len(parsed["summary"]),
        )
    
    def _parse_response(self, response: str) -> dict[str, Any]:
        """Parse structured response from LLM."""
        result = {
            "summary": "",
            "signals": [],
            "entities": [],
            "sentiment": None,
        }
        
        lines = response.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("Summary:"):
                result["summary"] = line[8:].strip()
            elif line.startswith("Signals:"):
                signals = line[8:].strip()
                result["signals"] = [s.strip() for s in signals.split(",") if s.strip()]
            elif line.startswith("Entities:"):
                entities = line[9:].strip()
                result["entities"] = [e.strip() for e in entities.split(",") if e.strip()]
            elif line.startswith("Sentiment:"):
                result["sentiment"] = line[10:].strip().lower()
        
        # Fallback if parsing fails
        if not result["summary"]:
            result["summary"] = response[:500]
        
        return result


class AnalysisSummarizer(Summarizer):
    """Summarizer for analysis outputs and insights.
    
    Preserves:
    - Conclusions and recommendations
    - Supporting evidence
    - Confidence levels
    - Risk factors
    """
    
    SYSTEM_PROMPT = """You are an analysis summarizer for investment insights. Create LOSSLESS summaries that preserve critical analytical content.

RULES:
1. PRESERVE: conclusions, recommendations, confidence levels
2. KEEP: supporting evidence and reasoning
3. MAINTAIN: risk factors and caveats
4. EXTRACT: actionable insights
5. BE STRUCTURED: use clear categorization

OUTPUT FORMAT:
Conclusion: [main conclusion]
Evidence: [key supporting points]
Risks: [risk factors]
Action: [recommended action if any]
Confidence: [high/medium/low]"""

    async def summarize(
        self,
        content: str,
        context: dict[str, Any] | None = None,
    ) -> SummaryResult:
        """Summarize analysis while preserving conclusions and evidence."""
        model = self._get_model()
        
        user_prompt = f"Summarize this investment analysis:\n\n{content}"
        
        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
        
        response = await model.ainvoke(messages)
        parsed = self._parse_response(response.content)
        
        return SummaryResult(
            summary=parsed["summary"],
            key_signals=parsed["signals"],
            entities=[],
            sentiment=parsed.get("confidence"),
            original_length=len(content),
            summary_length=len(parsed["summary"]),
        )
    
    def _parse_response(self, response: str) -> dict[str, Any]:
        """Parse structured response from LLM."""
        result = {
            "summary": "",
            "signals": [],
            "confidence": None,
        }
        
        parts = []
        lines = response.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("Conclusion:"):
                parts.append(line[11:].strip())
            elif line.startswith("Action:"):
                action = line[7:].strip()
                if action and action.lower() != "none":
                    result["signals"].append(f"Action: {action}")
            elif line.startswith("Risks:"):
                risks = line[6:].strip()
                if risks:
                    result["signals"].append(f"Risks: {risks}")
            elif line.startswith("Confidence:"):
                result["confidence"] = line[11:].strip().lower()
        
        result["summary"] = " ".join(parts) if parts else response[:500]
        
        return result


class TrendSummarizer(Summarizer):
    """Summarizer for trend analysis and macro insights.
    
    Preserves:
    - Trend identification
    - Direct and indirect beneficiaries
    - Causal chains
    - Time horizons
    """
    
    SYSTEM_PROMPT = """You are a trend analyst summarizer. Create LOSSLESS summaries of macro trends and their investment implications.

RULES:
1. IDENTIFY: the core trend clearly
2. MAP: direct beneficiaries (obvious plays)
3. DISCOVER: indirect beneficiaries (non-obvious plays)
4. TRACE: causal chain (A → B → C)
5. ESTIMATE: time horizon and confidence

OUTPUT FORMAT:
Trend: [name and description]
Direct: [obvious beneficiaries]
Indirect: [non-obvious beneficiaries]
Chain: [causal reasoning]
Horizon: [time estimate]"""

    async def summarize(
        self,
        content: str,
        context: dict[str, Any] | None = None,
    ) -> SummaryResult:
        """Summarize trend analysis preserving causal chains."""
        model = self._get_model()
        
        user_prompt = f"Analyze this for investment trends:\n\n{content}"
        
        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
        
        response = await model.ainvoke(messages)
        
        # For trends, we keep more of the response as it's inherently analytical
        return SummaryResult(
            summary=response.content[:1000],
            key_signals=[],
            entities=[],
            sentiment=None,
            original_length=len(content),
            summary_length=len(response.content[:1000]),
        )
