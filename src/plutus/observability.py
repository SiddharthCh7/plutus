"""Observability setup using Langfuse."""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

from plutus.config import get_settings
from plutus.logging import get_logger

logger = get_logger(__name__)

# Langfuse client (lazy init)
_langfuse = None


def get_langfuse():
    """Get Langfuse client (lazy initialization)."""
    global _langfuse
    
    if _langfuse is not None:
        return _langfuse
    
    settings = get_settings()
    
    if not settings.langfuse.is_configured:
        logger.warning("Langfuse not configured")
        return None
    
    try:
        from langfuse import Langfuse
        
        _langfuse = Langfuse(
            public_key=settings.langfuse.public_key,
            secret_key=settings.langfuse.secret_key,
            host=settings.langfuse.host,
        )
        
        logger.info("Langfuse initialized")
        
    except ImportError:
        logger.warning("Langfuse package not installed")
        return None
    except Exception as e:
        logger.error("Failed to initialize Langfuse", error=str(e))
        return None
    
    return _langfuse


def trace_agent(agent_name: str):
    """Decorator to trace agent execution with Langfuse.
    
    Usage:
        @trace_agent("news_monitor")
        async def run(self, context):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            langfuse = get_langfuse()
            
            if langfuse is None:
                return await func(*args, **kwargs)
            
            # Create trace
            trace = langfuse.trace(
                name=f"agent:{agent_name}",
                metadata={"agent": agent_name},
            )
            
            try:
                result = await func(*args, **kwargs)
                
                trace.update(
                    output=result,
                    level="DEFAULT",
                )
                
                return result
                
            except Exception as e:
                trace.update(
                    output={"error": str(e)},
                    level="ERROR",
                )
                raise
            finally:
                langfuse.flush()
        
        return wrapper
    return decorator


def trace_llm_call(model: str, prompt: str, response: str, tokens: int | None = None):
    """Log an LLM call to Langfuse."""
    langfuse = get_langfuse()
    
    if langfuse is None:
        return
    
    try:
        langfuse.generation(
            name="llm_call",
            model=model,
            input=prompt,
            output=response,
            usage={"total_tokens": tokens} if tokens else None,
        )
    except Exception as e:
        logger.warning("Failed to log to Langfuse", error=str(e))


def shutdown_observability():
    """Shutdown Langfuse (flush pending data)."""
    global _langfuse
    
    if _langfuse is not None:
        try:
            _langfuse.flush()
            _langfuse.shutdown()
            logger.info("Langfuse shutdown complete")
        except Exception as e:
            logger.warning("Error during Langfuse shutdown", error=str(e))
        
        _langfuse = None
