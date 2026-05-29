from __future__ import annotations

import logging
import sys
from pathlib import Path

from config import BASE_DIR


LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

try:
    from loguru import logger as _logger

    _logger.remove()
    _logger.add(
        sys.stderr,
        level="INFO",
        enqueue=True,
        backtrace=True,
        diagnose=False,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level}</level> | {message}",
    )
    _logger.add(
        LOG_DIR / "crypto_ai_bot.log",
        rotation="25 MB",
        retention="30 days",
        compression="zip",
        serialize=True,
        enqueue=True,
        backtrace=True,
        diagnose=False,
    )
    logger = _logger
except Exception:  # pragma: no cover - fallback for minimal test envs
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stderr), logging.FileHandler(LOG_DIR / "crypto_ai_bot.log")],
    )
    logger = logging.getLogger("crypto_ai_bot")
