from __future__ import annotations

import asyncio
from typing import Any

import requests

from config import Settings, settings
from utils.logger import logger

try:
    import aiohttp
except Exception:  # pragma: no cover
    aiohttp = None


class AlertManager:
    def __init__(self, cfg: Settings = settings) -> None:
        self.settings = cfg

    async def send(self, message: str, context: dict[str, Any] | None = None) -> None:
        if not self.settings.telegram_bot_token or not self.settings.telegram_chat_id:
            logger.warning(f"Alert: {message} | {context or {}}")
            return
        url = f"https://api.telegram.org/bot{self.settings.telegram_bot_token}/sendMessage"
        payload = {"chat_id": self.settings.telegram_chat_id, "text": f"{message}\n{context or {}}"}
        try:
            if aiohttp:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                    async with session.post(url, json=payload) as response:
                        response.raise_for_status()
            else:
                await asyncio.to_thread(lambda: requests.post(url, json=payload, timeout=10).raise_for_status())
        except Exception as exc:
            logger.warning(f"Failed to send alert: {exc}")
