"""Graph nodes - Agent implementations as workflow nodes.

Each node wraps an agent's execution and updates state.
Designed for context efficiency - agents receive only what they need.
"""

from __future__ import annotations

from plutus.graph.state import PlutusState
from plutus.logging import get_logger, LogContext

logger = get_logger(__name__)


async def coordinator_node(state: PlutusState) -> dict:
    """Coordinator node - orchestrates agent execution.
    
    Responsibilities:
    - Tracks which agents have run
    - Determines next agent based on task
    - Aggregates insights for final report
    """
    with LogContext(agent="coordinator", task=state.task_type):
        logger.info("Coordinator evaluating state")
        
        # First invocation - initialize
        if not state.agents_invoked:
            logger.info(
                "Starting task",
                task_type=state.task_type,
                tickers=state.portfolio_tickers,
            )
        
        return {}  # Routing handled by edges


async def portfolio_tracker_node(state: PlutusState) -> dict:
    """Portfolio Tracker - monitors holdings and calculates P&L.
    
    Context: Portfolio summary + market snapshot
    Output: Updated market data, P&L calculations
    """
    with LogContext(agent="portfolio_tracker"):
        logger.info("Tracking portfolio")
        
        # Import here to avoid circular imports
        from plutus.agents.portfolio_tracker import PortfolioTrackerAgent
        
        agent = PortfolioTrackerAgent()
        context = state.get_context_for_agent("portfolio_tracker")
        
        try:
            result = await agent.run(context)
            
            return {
                "market_snapshot": result.get("market_snapshot", {}),
                "agents_invoked": state.agents_invoked + ["portfolio_tracker"],
            }
        except Exception as e:
            logger.error("Portfolio tracker failed", error=str(e))
            return {
                "errors": state.errors + [f"portfolio_tracker: {e}"],
                "agents_invoked": state.agents_invoked + ["portfolio_tracker"],
            }


async def news_monitor_node(state: PlutusState) -> dict:
    """News Monitor - fetches and analyzes relevant news.
    
    Context: Portfolio tickers
    Output: News digest, top news items
    """
    with LogContext(agent="news_monitor"):
        logger.info("Monitoring news")
        
        from plutus.agents.news_monitor import NewsMonitorAgent
        
        agent = NewsMonitorAgent()
        context = state.get_context_for_agent("news_monitor")
        
        try:
            result = await agent.run(context)
            
            return {
                "news_digest": result.get("news_digest", ""),
                "top_news_items": result.get("top_news_items", []),
                "agents_invoked": state.agents_invoked + ["news_monitor"],
            }
        except Exception as e:
            logger.error("News monitor failed", error=str(e))
            return {
                "errors": state.errors + [f"news_monitor: {e}"],
                "agents_invoked": state.agents_invoked + ["news_monitor"],
            }


async def market_analyst_node(state: PlutusState) -> dict:
    """Market Analyst - analyzes prices and technicals.
    
    Context: Portfolio + market snapshot
    Output: Technical analysis, price insights
    """
    with LogContext(agent="market_analyst"):
        logger.info("Analyzing market")
        
        from plutus.agents.market_analyst import MarketAnalystAgent
        
        agent = MarketAnalystAgent()
        context = state.get_context_for_agent("market_analyst")
        
        try:
            result = await agent.run(context)
            
            return {
                "market_snapshot": {
                    **state.market_snapshot,
                    **result.get("analysis", {}),
                },
                "agents_invoked": state.agents_invoked + ["market_analyst"],
            }
        except Exception as e:
            logger.error("Market analyst failed", error=str(e))
            return {
                "errors": state.errors + [f"market_analyst: {e}"],
                "agents_invoked": state.agents_invoked + ["market_analyst"],
            }


async def risk_assessor_node(state: PlutusState) -> dict:
    """Risk Assessor - generates exit signals.
    
    Context: News digest + market data + portfolio
    Output: Risk signals (EXIT, REDUCE, HOLD, ACCUMULATE)
    """
    with LogContext(agent="risk_assessor"):
        logger.info("Assessing risks")
        
        from plutus.agents.risk_assessor import RiskAssessorAgent
        
        agent = RiskAssessorAgent()
        context = state.get_context_for_agent("risk_assessor")
        
        try:
            result = await agent.run(context)
            
            signals = result.get("risk_signals", [])
            should_notify = any(
                s.get("signal") in ["exit", "reduce", "accumulate"] 
                for s in signals
            )
            
            return {
                "risk_signals": signals,
                "should_notify": should_notify or state.should_notify,
                "agents_invoked": state.agents_invoked + ["risk_assessor"],
            }
        except Exception as e:
            logger.error("Risk assessor failed", error=str(e))
            return {
                "errors": state.errors + [f"risk_assessor: {e}"],
                "agents_invoked": state.agents_invoked + ["risk_assessor"],
            }


async def opportunity_scout_node(state: PlutusState) -> dict:
    """Opportunity Scout - finds undervalued stocks.
    
    Context: News + trends + market data
    Output: Investment opportunities
    """
    with LogContext(agent="opportunity_scout"):
        logger.info("Scouting opportunities")
        
        from plutus.agents.opportunity_scout import OpportunityScoutAgent
        
        agent = OpportunityScoutAgent()
        context = state.get_context_for_agent("opportunity_scout")
        
        try:
            result = await agent.run(context)
            
            opportunities = result.get("opportunities", [])
            should_notify = len(opportunities) > 0
            
            return {
                "opportunities": opportunities,
                "should_notify": should_notify or state.should_notify,
                "agents_invoked": state.agents_invoked + ["opportunity_scout"],
            }
        except Exception as e:
            logger.error("Opportunity scout failed", error=str(e))
            return {
                "errors": state.errors + [f"opportunity_scout: {e}"],
                "agents_invoked": state.agents_invoked + ["opportunity_scout"],
            }


async def trend_interpreter_node(state: PlutusState) -> dict:
    """Trend Interpreter - decodes macro trends.
    
    The "Intelligent Investor" logic:
    - Identifies trends (e.g., AI boom)
    - Maps direct beneficiaries (NVIDIA)
    - Discovers indirect beneficiaries (TSMC, data centers)
    
    Context: News digest
    Output: Trend insights with beneficiary chains
    """
    with LogContext(agent="trend_interpreter"):
        logger.info("Interpreting trends")
        
        from plutus.agents.trend_interpreter import TrendInterpreterAgent
        
        agent = TrendInterpreterAgent()
        context = state.get_context_for_agent("trend_interpreter")
        
        try:
            result = await agent.run(context)
            
            return {
                "trend_insights": result.get("trend_insights", []),
                "agents_invoked": state.agents_invoked + ["trend_interpreter"],
            }
        except Exception as e:
            logger.error("Trend interpreter failed", error=str(e))
            return {
                "errors": state.errors + [f"trend_interpreter: {e}"],
                "agents_invoked": state.agents_invoked + ["trend_interpreter"],
            }


async def report_generator_node(state: PlutusState) -> dict:
    """Report Generator - creates consolidated report.
    
    Generates a detailed, actionable report with:
    - Portfolio summary with P&L
    - Risk signals with call to action
    - Expected growth indicators
    - Market analysis
    - Opportunities
    """
    with LogContext(agent="report_generator"):
        logger.info("Generating report")
        
        sections = []
        
        # Portfolio summary with market data
        if state.portfolio_summary:
            portfolio_section = f"## 📊 Portfolio\n{state.portfolio_summary}"
            
            # Add market data if available
            if state.market_snapshot:
                market_lines = []
                for ticker, data in state.market_snapshot.items():
                    if isinstance(data, dict) and data.get("price"):
                        change = data.get("change_percent", 0)
                        change_icon = "🟢" if change >= 0 else "🔴"
                        market_lines.append(
                            f"  {ticker}: {data['price']:.2f} {change_icon} {change:+.2f}%"
                        )
                if market_lines:
                    portfolio_section += "\n\n### Current Prices\n" + "\n".join(market_lines)
            
            sections.append(portfolio_section)
        
        # Risk signals with call to action and expected growth
        if state.risk_signals:
            signals_lines = []
            for s in state.risk_signals:
                signal = s.get('signal', 'hold').upper()
                ticker = s.get('ticker')
                reason = s.get('primary_reason', 'No reason')
                confidence = s.get('confidence', 0.5)
                
                # Signal icons
                if signal == "EXIT":
                    icon = "🚨"
                    action = "SELL immediately"
                    growth = "⬇️ Negative outlook"
                elif signal == "REDUCE":
                    icon = "⚠️"
                    action = "Consider reducing position by 25-50%"
                    growth = "⬇️ Cautious outlook"
                elif signal == "ACCUMULATE":
                    icon = "✅"
                    action = "Consider increasing position"
                    growth = "⬆️ Positive outlook"
                else:
                    icon = "➖"
                    action = "Maintain current position"
                    growth = "➡️ Neutral outlook"
                
                # Find relevant news
                relevant_news = []
                for item in state.top_news_items:
                    related = [t.upper() for t in item.get("related_tickers", [])]
                    if ticker.upper() in related:
                        relevant_news.append(item)
                
                news_section = ""
                if relevant_news:
                    news_section = "\n**Latest News:**\n"
                    for news in relevant_news[:2]:
                        summary = news.get("summary", "")
                        if len(summary) > 150:
                            summary = summary[:147] + "..."
                        news_section += f"- {summary}\n"
                
                signals_lines.append(
                    f"### {icon} {ticker}: {signal}\n"
                    f"**Reason:** {reason}\n"
                    f"**Confidence:** {confidence:.0%}\n"
                    f"**Expected Growth:** {growth}\n"
                    f"**📌 Call to Action:** {action}\n"
                    f"{news_section}"
                )
            
            sections.append("## ⚡ Risk Signals\n" + "\n\n".join(signals_lines))
        
        # Opportunities with investment thesis
        if state.opportunities:
            opps_lines = []
            for o in state.opportunities:
                ticker = o.get('ticker')
                company = o.get('company_name', ticker)
                sector = o.get('sector', 'Unknown')
                thesis = o.get('thesis', 'N/A')
                score = o.get('opportunity_score', 0.5)
                
                opps_lines.append(
                    f"### 💡 {ticker} ({company})\n"
                    f"**Sector:** {sector}\n"
                    f"**Score:** {score:.0%}\n"
                    f"**Thesis:** {thesis}\n"
                    f"**📌 Call to Action:** Research further, consider entry if thesis holds"
                )
            
            sections.append("## 🎯 Opportunities\n" + "\n\n".join(opps_lines))
        
        # Trend insights
        if state.trend_insights:
            trends_lines = []
            for t in state.trend_insights:
                trend = t.get('trend_name', 'Unknown')
                desc = t.get('description', '')
                direct = ', '.join(t.get('direct_beneficiaries', [])[:3])
                indirect = ', '.join(t.get('indirect_beneficiaries', [])[:3])
                reasoning = t.get('reasoning', '')
                
                trends_lines.append(
                    f"### 📈 {trend}\n"
                    f"{desc}\n"
                    f"**Direct plays:** {direct or 'N/A'}\n"
                    f"**Indirect plays:** {indirect or 'N/A'}\n"
                    f"**Reasoning:** {reasoning[:200]}"
                )
            
            sections.append("## 🌊 Trend Insights\n" + "\n\n".join(trends_lines))
        
        # News summary
        if state.news_digest:
            sections.append(f"## 📰 News Summary\n{state.news_digest}")
        else:
            sections.append("## 📰 News Summary\nNo relevant news found.")
        
        # Errors
        if state.errors:
            sections.append("## ❌ Errors\n" + "\n".join(f"- {e}" for e in state.errors))
        
        report = "\n\n".join(sections) if sections else "No analysis results."
        
        logger.info("Report generated", length=len(report))
        
        return {
            "final_report": report,
        }


async def notifier_node(state: PlutusState) -> dict:
    """Notifier - sends alerts via configured channels.
    
    Currently: Email
    Future: Telegram, push notifications, etc.
    """
    with LogContext(agent="notifier"):
        logger.info("Sending notification")
        
        from plutus.notifications import get_notifier
        
        try:
            notifier = get_notifier()
            await notifier.send(
                subject=f"Plutus Alert: {state.task_type}",
                body=state.final_report,
            )
            logger.info("Notification sent")
        except Exception as e:
            logger.error("Notification failed", error=str(e))
            return {
                "warnings": state.warnings + [f"Notification failed: {e}"],
            }
        
        return {}
