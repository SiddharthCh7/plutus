"""Portfolio data models."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class Holding(BaseModel):
    """A single stock/fund holding in the portfolio."""
    
    ticker: str = Field(..., description="Stock/fund ticker symbol")
    buy_price: float = Field(..., description="Purchase price per unit")
    quantity: int = Field(..., description="Number of units held")
    buy_date: date | None = Field(None, description="Date of purchase")
    
    # Optional enrichment fields
    sector: str | None = Field(None, description="Sector classification")
    notes: str | None = Field(None, description="Investment thesis")
    target_price: float | None = Field(None, description="Target exit price")
    
    # Computed at runtime (not persisted)
    current_price: float | None = Field(None, exclude=True)
    
    @property
    def investment_value(self) -> float:
        """Total investment value at buy price."""
        return self.buy_price * self.quantity
    
    @property
    def current_value(self) -> float | None:
        """Current market value (if price available)."""
        if self.current_price is None:
            return None
        return self.current_price * self.quantity
    
    @property
    def pnl(self) -> float | None:
        """Profit/Loss in absolute terms."""
        if self.current_price is None:
            return None
        return (self.current_price - self.buy_price) * self.quantity
    
    @property
    def pnl_percent(self) -> float | None:
        """Profit/Loss as percentage."""
        if self.current_price is None:
            return None
        return ((self.current_price - self.buy_price) / self.buy_price) * 100


class Portfolio(BaseModel):
    """User's complete portfolio."""
    
    holdings: list[Holding] = Field(default_factory=list)
    
    @property
    def total_investment(self) -> float:
        """Total invested amount."""
        return sum(h.investment_value for h in self.holdings)
    
    @property
    def total_current_value(self) -> float | None:
        """Total current market value."""
        values = [h.current_value for h in self.holdings]
        if any(v is None for v in values):
            return None
        return sum(v for v in values if v is not None)
    
    @property
    def tickers(self) -> list[str]:
        """List of all ticker symbols."""
        return [h.ticker for h in self.holdings]
    
    def get_holding(self, ticker: str) -> Holding | None:
        """Get holding by ticker."""
        for h in self.holdings:
            if h.ticker.upper() == ticker.upper():
                return h
        return None
    
    @classmethod
    def from_json_file(cls, path: Path) -> "Portfolio":
        """Load portfolio from JSON file."""
        import json
        with open(path) as f:
            data = json.load(f)
        return cls(**data)
    
    def to_json_file(self, path: Path) -> None:
        """Save portfolio to JSON file."""
        import json
        with open(path, "w") as f:
            json.dump(self.model_dump(mode="json"), f, indent=2, default=str)
    
    def to_summary(self) -> str:
        """Generate concise text summary for LLM context.
        
        This is part of context engineering - only essential info.
        """
        lines = [f"Portfolio: {len(self.holdings)} holdings"]
        for h in self.holdings:
            pnl_str = f"{h.pnl_percent:+.1f}%" if h.pnl_percent else "N/A"
            lines.append(f"  {h.ticker}: {h.quantity} @ {h.buy_price} ({pnl_str})")
        return "\n".join(lines)
