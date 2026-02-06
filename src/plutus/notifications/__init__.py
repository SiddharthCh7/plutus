"""Notifications package - Modular alerting system."""

from plutus.notifications.base import Notifier
from plutus.notifications.email import EmailNotifier


def get_notifier() -> Notifier:
    """Get the configured notifier.
    
    Currently returns email notifier.
    Future: Support multiple channels (Telegram, etc.)
    """
    return EmailNotifier()


__all__ = ["Notifier", "EmailNotifier", "get_notifier"]
