# Plutus - AI-Powered Investment Research & Portfolio Monitoring Assistant

A multi-agent AI system for intelligent investment analysis, portfolio monitoring, and opportunity discovery.

## Features

- 📊 **Portfolio Tracking** - Monitor holdings from JSON file
- 📰 **News Intelligence** - Real-time news analysis for market insights
- ⚠️ **Exit Signals** - Predict potential falls based on news, prices, fundamentals
- 🔍 **Opportunity Discovery** - Find undervalued stocks with long-term potential
- 🧠 **Trend Analysis** - Smart decoding of macro trends and indirect beneficiaries

## Architecture

Uses LangGraph for multi-agent orchestration with:
- **Coordinator Agent** - Orchestrates specialist agents
- **Portfolio Tracker** - Monitors holdings and P&L
- **News Monitor** - Fetches and filters relevant news
- **Market Analyst** - Analyzes prices and technicals
- **Risk Assessor** - Generates exit signals
- **Opportunity Scout** - Finds undervalued stocks
- **Trend Interpreter** - Decodes macro trends

## Quick Start

```bash
# Install dependencies
uv sync

# Set up environment
cp .env.example .env
# Edit .env with your API keys

# Run Plutus
uv run plutus
```

## Configuration

- `config/models.yaml` - LLM provider configuration
- `config/schedule.yaml` - Task scheduling (cron)
- `data/portfolio.json` - Your holdings

## Tech Stack

- **LangGraph** - Multi-agent orchestration
- **Qdrant** - Vector storage for context
- **Langfuse** - Observability & tracing
- **Yahoo Finance** - Market data
- **structlog** - Structured logging

