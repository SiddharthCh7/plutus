"""Trading signals and insights models."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class SignalType(str, Enum):
    """Type of trading signal."""
    
    EXIT = "exit"           # Recommend selling
    REDUCE = "reduce"       # Reduce position
    HOLD = "hold"           # Maintain position
    ACCUMULATE = "accumulate"  # Buy more


class RiskSignal(BaseModel):
    """Risk assessment signal for a holding."""
    
    ticker: str = Field(..., description="Stock ticker")
    signal: SignalType = Field(..., description="Recommended action")
    confidence: float = Field(..., ge=0, le=1, description="Confidence 0-1")
    
    # Reasoning (concise for context)
    primary_reason: str = Field(..., description="Main reason for signal")
    supporting_factors: list[str] = Field(default_factory=list)
    
    # Metadata
    generated_at: datetime = Field(default_factory=datetime.now)
    news_based: bool = Field(False, description="Signal from news analysis")
    price_based: bool = Field(False, description="Signal from price analysis")
    
    def to_context_string(self) -> str:
        """Concise string for LLM context."""
        return (
            f"{self.ticker}: {self.signal.value.upper()} "
            f"(confidence: {self.confidence:.0%}) - {self.primary_reason}"
        )


class Opportunity(BaseModel):
    """Investment opportunity identified by the system."""
    
    ticker: str = Field(..., description="Stock ticker")
    company_name: str = Field("", description="Company name")
    sector: str = Field("", description="Sector")
    
    # Valuation
    current_price: float | None = Field(None)
    fair_value_estimate: float | None = Field(None)
    upside_potential: float | None = Field(None, description="Percentage upside")
    
    # Reasoning
    thesis: str = Field(..., description="Investment thesis")
    catalysts: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    
    # Scoring
    opportunity_score: float = Field(..., ge=0, le=1)
    time_horizon: str = Field("long_term", description="Investment horizon")
    
    # Metadata
    discovered_at: datetime = Field(default_factory=datetime.now)
    source: str = Field("", description="How it was discovered")
    
    def to_context_string(self) -> str:
        """Concise string for LLM context."""
        upside = f"+{self.upside_potential:.0%}" if self.upside_potential else "TBD"
        return (
            f"{self.ticker} ({self.sector}): {upside} potential | "
            f"Score: {self.opportunity_score:.0%} | {self.thesis[:100]}"
        )


class TrendInsight(BaseModel):
    """Macro trend insight with indirect beneficiaries.
    
    Example: AI boom → semiconductor demand → TSMC, ASML benefit
    """
    
    trend_name: str = Field(..., description="Name of the trend")
    description: str = Field(..., description="What's happening")
    
    # Direct and indirect beneficiaries (The Intelligent Investor logic)
    direct_beneficiaries: list[str] = Field(
        default_factory=list,
        description="Obvious beneficiaries",
    )
    indirect_beneficiaries: list[str] = Field(
        default_factory=list,
        description="Non-obvious beneficiaries",
    )
    
    # Reasoning chain
    reasoning: str = Field(
        ...,
        description="How indirect beneficiaries are connected",
    )
    
    # Scoring
    confidence: float = Field(..., ge=0, le=1)
    time_to_impact: str = Field("", description="When impact expected")
    
    # Metadata
    identified_at: datetime = Field(default_factory=datetime.now)
    news_sources: list[str] = Field(default_factory=list)
    
    def to_context_string(self) -> str:
        """Concise string for LLM context."""
        indirect = ", ".join(self.indirect_beneficiaries[:3])
        return (
            f"Trend: {self.trend_name} | "
            f"Indirect plays: {indirect} | "
            f"Confidence: {self.confidence:.0%}"
        )
