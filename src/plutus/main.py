"""Plutus CLI - Main entry point."""

from __future__ import annotations

import asyncio
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from plutus import __version__
from plutus.config import get_settings, get_schedule_config
from plutus.graph import PlutusState, get_workflow
from plutus.logging import setup_logging, get_logger
from plutus.models import Portfolio
from plutus.observability import shutdown_observability
from plutus.scheduler import get_scheduler

# Initialize CLI
app = typer.Typer(
    name="plutus",
    help="Personal Financial Portfolio Manager - Multi-Agent AI System",
    add_completion=False,
)
console = Console()

# Setup logging on import
setup_logging()
logger = get_logger(__name__)


@app.command()
def version():
    """Show version information."""
    console.print(f"[bold green]Plutus[/bold green] v{__version__}")


@app.command()
def run(
    task: str = typer.Argument(
        "portfolio_monitor",
        help="Task to run: portfolio_monitor, opportunity_discovery, news_digest",
    ),
    once: bool = typer.Option(
        False,
        "--once",
        "-o",
        help="Run once instead of scheduling",
    ),
):
    """Run Plutus tasks.
    
    Examples:
        plutus run portfolio_monitor --once
        plutus run opportunity_discovery
    """
    console.print(Panel(
        f"[bold]Running task:[/bold] {task}",
        title="Plutus",
        border_style="green",
    ))
    
    if once:
        asyncio.run(_run_task_once(task))
    else:
        _run_scheduled(task)


async def _run_task_once(task_name: str):
    """Run a single task execution."""
    # Import here to avoid circular imports
    from plutus.logging import suppress_logs, restore_logs
    
    settings = get_settings()
    
    # Load portfolio
    try:
        portfolio = Portfolio.from_json_file(settings.portfolio_path)
        console.print(f"[green]Loaded portfolio:[/green] {len(portfolio.holdings)} holdings")
    except Exception as e:
        console.print(f"[red]Failed to load portfolio:[/red] {e}")
        portfolio = Portfolio()
    
    # Initialize state
    initial_state = PlutusState(
        task_type=task_name,
        portfolio_summary=portfolio.to_summary(),
        portfolio_tickers=portfolio.tickers,
    )
    
    # Suppress logs before compiling workflow for clean output
    console.print("[yellow]Executing workflow...[/yellow]")
    suppress_logs()
    
    # Get workflow (may log on first compile)
    workflow = get_workflow()
    
    try:
        # Suppress verbose logs during execution
        suppress_logs()
        
        result = await workflow.ainvoke(initial_state)
        
        # Restore logs before printing
        restore_logs()
        
        # Display report
        console.print()
        console.print(Panel(
            result.get("final_report", "No report generated"),
            title="Analysis Report",
            border_style="blue",
        ))
        
        # Log errors if any
        if result.get("errors"):
            console.print("[red]Errors:[/red]")
            for error in result["errors"]:
                console.print(f"  - {error}")
        
    except Exception as e:
        restore_logs()
        console.print(f"[red]Workflow failed:[/red] {e}")
        logger.error("Workflow execution failed", error=str(e))
    finally:
        shutdown_observability()


def _run_scheduled(initial_task: str):
    """Run with scheduler."""
    console.print("[yellow]Starting scheduler...[/yellow]")
    
    scheduler = get_scheduler()
    
    # Register task handler
    async def task_handler(task_name: str, agents: list[str], **kwargs):
        await _run_task_once(task_name)
    
    scheduler.register_handler("portfolio_monitor", task_handler)
    scheduler.register_handler("opportunity_discovery", task_handler)
    scheduler.register_handler("news_digest", task_handler)
    
    # Schedule all tasks
    scheduler.schedule_tasks()
    
    # Show scheduled tasks
    table = Table(title="Scheduled Tasks")
    table.add_column("Task", style="cyan")
    table.add_column("Next Run", style="green")
    
    for job in scheduler.list_scheduled():
        table.add_row(job["id"], job["next_run"] or "Not scheduled")
    
    console.print(table)
    
    # Run initial task
    console.print(f"[yellow]Running initial task: {initial_task}[/yellow]")
    asyncio.run(_run_task_once(initial_task))
    
    # Start scheduler (blocks)
    try:
        scheduler.start()
        console.print("[green]Scheduler running. Press Ctrl+C to stop.[/green]")
        
        # Keep running
        import signal
        signal.pause()
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping scheduler...[/yellow]")
        scheduler.stop()
        shutdown_observability()


@app.command()
def portfolio():
    """Show current portfolio."""
    settings = get_settings()
    
    try:
        p = Portfolio.from_json_file(settings.portfolio_path)
    except Exception as e:
        console.print(f"[red]Failed to load portfolio:[/red] {e}")
        return
    
    table = Table(title="Portfolio Holdings")
    table.add_column("Ticker", style="cyan")
    table.add_column("Quantity", justify="right")
    table.add_column("Buy Price", justify="right")
    table.add_column("Sector")
    
    for h in p.holdings:
        table.add_row(
            h.ticker,
            str(h.quantity),
            f"${h.buy_price:.2f}" if h.buy_price else "N/A",
            h.sector or "-",
        )
    
    console.print(table)
    console.print(f"\n[green]Total Investment:[/green] ${p.total_investment:,.2f}")


@app.command()
def schedule():
    """Show scheduled tasks."""
    config = get_schedule_config()
    
    table = Table(title="Task Schedule")
    table.add_column("Task", style="cyan")
    table.add_column("Enabled", justify="center")
    table.add_column("Start", justify="right")
    table.add_column("Frequency")
    table.add_column("Agents")
    
    for name, task in config.tasks.items():
        enabled = "✅" if task.get("enabled", True) else "❌"
        agents = ", ".join(task.get("agents", [])[:3])
        if len(task.get("agents", [])) > 3:
            agents += "..."
        
        table.add_row(
            name,
            enabled,
            task.get("start_time", "-"),
            f"{task.get('frequency_mins', 60)} min",
            agents,
        )
    
    console.print(table)


@app.command()
def research(
    ticker: str = typer.Argument(
        ...,
        help="Stock ticker to research (e.g., AAPL, RELIANCE.NS)",
    ),
):
    """Perform deep research on a single ticker.
    
    Analyzes fundamentals, growth, valuation, red flags, and provides
    an investment verdict with call to action.
    
    Examples:
        plutus research AAPL
        plutus research KELLTONTEC.NS
    """
    console.print(Panel(
        f"[bold]Deep Research:[/bold] {ticker}",
        title="Plutus Research",
        border_style="green",
    ))
    
    asyncio.run(_run_research(ticker))


async def _run_research(ticker: str):
    """Run deep research on a ticker."""
    from plutus.agents.deep_research import DeepResearchAgent
    from plutus.logging import suppress_logs, restore_logs
    
    console.print(f"[yellow]Analyzing {ticker}...[/yellow]")
    console.print("[dim]Fetching fundamentals, financials, and market data...[/dim]")
    
    try:
        suppress_logs()
        
        agent = DeepResearchAgent()
        result = await agent.run({"ticker": ticker})
        
        restore_logs()
        
        # Display report
        console.print()
        console.print(Panel(
            result.get("research_report", "No report generated"),
            title=f"Research Report: {ticker}",
            border_style="blue",
        ))
        
    except Exception as e:
        restore_logs()
        console.print(f"[red]Research failed:[/red] {e}")


if __name__ == "__main__":
    app()
