"""Email notifier implementation."""

from __future__ import annotations

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import aiosmtplib

from plutus.config import get_settings
from plutus.logging import get_logger
from plutus.notifications.base import Notifier

logger = get_logger(__name__)


class EmailNotifier(Notifier):
    """Email notification implementation.
    
    Uses aiosmtplib for async email sending.
    Configured via environment variables.
    """
    
    def __init__(self) -> None:
        self._settings = get_settings()
    
    @property
    def is_configured(self) -> bool:
        """Check if email is properly configured."""
        return self._settings.email.is_configured
    
    async def send(
        self,
        subject: str,
        body: str,
        to_email: str | None = None,
        **kwargs,
    ) -> bool:
        """Send an email notification.
        
        Args:
            subject: Email subject
            body: Email body (plain text or markdown)
            to_email: Recipient (defaults to NOTIFICATION_EMAIL)
            
        Returns:
            True if sent successfully
        """
        if not self.is_configured:
            logger.warning("Email not configured, skipping notification")
            return False
        
        email_settings = self._settings.email
        recipient = to_email or self._settings.notification_email
        
        if not recipient:
            logger.warning("No recipient email configured")
            return False
        
        try:
            # Build message
            message = MIMEMultipart("alternative")
            message["Subject"] = f"[Plutus] {subject}"
            message["From"] = email_settings.user
            message["To"] = recipient
            
            # Add plain text body
            text_part = MIMEText(body, "plain")
            message.attach(text_part)
            
            # Add HTML body (simple conversion)
            html_body = self._markdown_to_html(body)
            html_part = MIMEText(html_body, "html")
            message.attach(html_part)
            
            # Send
            await aiosmtplib.send(
                message,
                hostname=email_settings.host,
                port=email_settings.port,
                username=email_settings.user,
                password=email_settings.password,
                start_tls=True,
            )
            
            logger.info(
                "Email sent",
                to=recipient,
                subject=subject,
            )
            
            return True
            
        except Exception as e:
            logger.error(
                "Failed to send email",
                error=str(e),
                to=recipient,
            )
            return False
    
    def _markdown_to_html(self, markdown: str) -> str:
        """Simple markdown to HTML conversion."""
        html = markdown
        
        # Headers
        lines = []
        for line in html.split("\n"):
            if line.startswith("## "):
                line = f"<h2>{line[3:]}</h2>"
            elif line.startswith("# "):
                line = f"<h1>{line[2:]}</h1>"
            elif line.startswith("- "):
                line = f"<li>{line[2:]}</li>"
            lines.append(line)
        
        html = "\n".join(lines)
        
        # Wrap in basic HTML
        return f"""<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; padding: 20px; }}
        h1, h2 {{ color: #333; }}
        li {{ margin: 5px 0; }}
    </style>
</head>
<body>
{html}
</body>
</html>"""
