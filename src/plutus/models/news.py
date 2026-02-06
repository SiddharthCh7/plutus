"""News data models."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class NewsSentiment(str, Enum):
    """Sentiment classification for news."""
    
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"


class NewsItem(BaseModel):
    """A single news item."""
    
    title: str = Field(..., description="News headline")
    source: str = Field(..., description="News source")
    url: str = Field(..., description="Article URL")
    published_at: datetime = Field(..., description="Publication timestamp")
    
    # Content (summarized, not raw)
    summary: str = Field("", description="LLM-generated summary")
    
    # Analysis results
    sentiment: NewsSentiment | None = Field(None, description="Sentiment")
    relevance_score: float = Field(0.0, description="0-1 relevance to portfolio")
    related_tickers: list[str] = Field(default_factory=list)
    
    # Categories
    categories: list[str] = Field(default_factory=list)
    is_market_moving: bool = Field(False, description="High-impact news")
    
    def to_context_string(self) -> str:
        """Generate concise string for LLM context.
        
        Context engineering: Only essential information.
        """
        sentiment_str = self.sentiment.value if self.sentiment else "unknown"
        tickers = ", ".join(self.related_tickers) if self.related_tickers else "general"
        
        return (
            f"[{self.published_at.strftime('%Y-%m-%d')}] "
            f"{self.title} | "
            f"Sentiment: {sentiment_str} | "
            f"Tickers: {tickers}"
        )


class NewsDigest(BaseModel):
    """Aggregated news digest - summarized for context efficiency."""
    
    generated_at: datetime = Field(default_factory=datetime.now)
    total_articles: int = Field(0)
    
    # Summarized content (not raw articles)
    market_summary: str = Field("", description="Overall market summary")
    portfolio_relevant: str = Field("", description="News relevant to holdings")
    opportunities: str = Field("", description="Potential opportunity signals")
    risks: str = Field("", description="Risk signals from news")
    
    # Key items only (not full list)
    top_items: list[NewsItem] = Field(
        default_factory=list,
        description="Top 5 most relevant items",
    )
    
    def to_context_string(self) -> str:
        """Generate concise digest for LLM context."""
        return f"""News Digest ({self.total_articles} articles analyzed):

Market: {self.market_summary}

Portfolio Impact: {self.portfolio_relevant}

Opportunities: {self.opportunities}

Risks: {self.risks}"""
