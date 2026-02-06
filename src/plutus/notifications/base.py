"""Abstract notifier interface - Modular and extensible."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Notifier(ABC):
    """Abstract notification interface.
    
    Implement for new channels:
    - EmailNotifier (current)
    - TelegramNotifier (future)
    - SlackNotifier (future)
    - PushNotifier (future)
    """
    
    @abstractmethod
    async def send(
        self,
        subject: str,
        body: str,
        **kwargs,
    ) -> bool:
        """Send a notification.
        
        Args:
            subject: Notification subject/title
            body: Notification content
            **kwargs: Channel-specific options
            
        Returns:
            True if sent successfully
        """
        pass
    
    @property
    @abstractmethod
    def is_configured(self) -> bool:
        """Check if notifier is properly configured."""
        pass
