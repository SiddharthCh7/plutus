"""Model Router - Multi-provider LLM selection.

Supports:
- Google Gemini (default)
- OpenAI
- Anthropic
- Future: OpenRouter, Ollama

Portable design for easy provider switching.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel

from plutus.config import get_model_config, get_settings
from plutus.logging import get_logger

logger = get_logger(__name__)


class ModelRouter:
    """Routes to appropriate LLM based on configuration.
    
    Supports per-agent model overrides and dynamic provider switching.
    """
    
    def __init__(self) -> None:
        self._settings = get_settings()
        self._model_config = get_model_config()
        self._models: dict[str, BaseChatModel] = {}
    
    def get_model(
        self,
        provider: str | None = None,
        model: str | None = None,
        agent_name: str | None = None,
        **kwargs: Any,
    ) -> BaseChatModel:
        """Get an LLM instance.
        
        Args:
            provider: Provider name (gemini, openai, anthropic)
            model: Model name (e.g., gemini-2.5-flash)
            agent_name: Agent requesting the model (for per-agent overrides)
            **kwargs: Additional model configuration
            
        Returns:
            Configured LLM instance
        """
        # Check for agent-specific override
        if agent_name:
            agent_model = self._model_config.get_agent_model(agent_name)
            if agent_model:
                model = agent_model
        
        # Use defaults if not specified
        provider = provider or self._model_config.default_provider
        model = model or self._model_config.default_model
        
        cache_key = f"{provider}:{model}"
        
        if cache_key not in self._models:
            self._models[cache_key] = self._create_model(provider, model, **kwargs)
            logger.info(
                "Created LLM instance",
                provider=provider,
                model=model,
                agent=agent_name,
            )
        
        return self._models[cache_key]
    
    def get_summarization_model(self) -> BaseChatModel:
        """Get the model configured for summarization.
        
        This can be overridden with fine-tuned models in the future.
        """
        config = self._model_config.summarization_config
        return self.get_model(
            provider=config.get("provider"),
            model=config.get("model"),
        )
    
    def _create_model(
        self,
        provider: str,
        model: str,
        **kwargs: Any,
    ) -> BaseChatModel:
        """Create a new LLM instance for the given provider."""
        if provider == "gemini":
            return self._create_gemini(model, **kwargs)
        elif provider == "openai":
            return self._create_openai(model, **kwargs)
        elif provider == "anthropic":
            return self._create_anthropic(model, **kwargs)
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def _create_gemini(self, model: str, **kwargs: Any) -> BaseChatModel:
        """Create a Gemini model instance."""
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=self._settings.google_api_key,
            temperature=kwargs.get("temperature", 0.7),
            **kwargs,
        )
    
    def _create_openai(self, model: str, **kwargs: Any) -> BaseChatModel:
        """Create an OpenAI model instance."""
        from langchain_openai import ChatOpenAI
        
        return ChatOpenAI(
            model=model,
            api_key=self._settings.openai_api_key,
            temperature=kwargs.get("temperature", 0.7),
            **kwargs,
        )
    
    def _create_anthropic(self, model: str, **kwargs: Any) -> BaseChatModel:
        """Create an Anthropic model instance."""
        from langchain_anthropic import ChatAnthropic
        
        return ChatAnthropic(
            model=model,
            api_key=self._settings.anthropic_api_key,
            temperature=kwargs.get("temperature", 0.7),
            **kwargs,
        )


# Singleton instance
_router: ModelRouter | None = None


def get_model_router() -> ModelRouter:
    """Get the model router singleton."""
    global _router
    if _router is None:
        _router = ModelRouter()
    return _router
