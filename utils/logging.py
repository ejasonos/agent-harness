from __future__ import annotations

import json
import logging
import os
import re
import sys
import threading
import traceback
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Any, Iterator


_LOGGER_NAME = "local_agent"
_LOG_CONTEXT: ContextVar[dict[str, Any]] = ContextVar(
	"local_agent_log_context",
	default={},
)
_CONFIG_LOCK = threading.Lock()
_SENSITIVE_KEY = re.compile(
	r"api[_-]?key|authorization|cookie|credential|password|secret|token",
	re.IGNORECASE,
)


def configure_logging(level: str | None = None) -> logging.Logger:
	logger = logging.getLogger(_LOGGER_NAME)
	configured_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
	numeric_level = getattr(logging, configured_level, logging.INFO)
	if not isinstance(numeric_level, int):
		numeric_level = logging.INFO

	with _CONFIG_LOCK:
		logger.setLevel(numeric_level)
		logger.propagate = False

		if not any(
			getattr(handler, "_local_agent_console", False)
			for handler in logger.handlers
		):
			console = logging.StreamHandler(sys.stderr)
			console.setFormatter(logging.Formatter("%(message)s"))
			console._local_agent_console = True
			logger.addHandler(console)

		log_file = os.getenv("LOG_FILE")
		if log_file and not any(
			getattr(handler, "baseFilename", None) == os.path.abspath(log_file)
			for handler in logger.handlers
		):
			os.makedirs(
				os.path.dirname(os.path.abspath(log_file)),
				exist_ok=True,
			)
			file_handler = RotatingFileHandler(
				log_file,
				maxBytes=10_000_000,
				backupCount=3,
				encoding="utf-8",
			)
			file_handler.setFormatter(logging.Formatter("%(message)s"))
			logger.addHandler(file_handler)

	return logger


def new_run_id() -> str:
	return uuid.uuid4().hex


@contextmanager
def log_context(**fields: Any) -> Iterator[None]:
	updated_context = {
		**_LOG_CONTEXT.get(),
		**fields,
	}
	token = _LOG_CONTEXT.set(updated_context)
	try:
		yield
	finally:
		_LOG_CONTEXT.reset(token)


def log_event(
	event: str,
	*,
	level: int = logging.INFO,
	exc_info: bool = False,
	**fields: Any,
) -> None:
	logger = configure_logging()
	if exc_info:
		fields["traceback"] = traceback.format_exc()
	entry = {
		"timestamp": datetime.now(timezone.utc).isoformat(
			timespec="milliseconds"
		),
		"event": event,
		**_LOG_CONTEXT.get(),
		**fields,
	}
	logger.log(
		level,
		json.dumps(_redact(entry), default=str, ensure_ascii=True),
	)


def _redact(value: Any) -> Any:
	if isinstance(value, dict):
		return {
			key: (
				"[REDACTED]"
				if _SENSITIVE_KEY.search(str(key))
				else _redact(item)
			)
			for key, item in value.items()
		}
	if isinstance(value, (list, tuple)):
		return [_redact(item) for item in value]
	return value
