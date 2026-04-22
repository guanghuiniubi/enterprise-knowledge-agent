import builtins
from importlib import import_module

from app.core.config import settings
from app.core.request_context import get_request_context
from app.security.redaction import redactor

std_logging = import_module("logging")
LOG_FORMAT = (
    "%(asctime)s %(levelname)s [%(name)s] "
    "[request_id=%(request_id)s session_id=%(session_id)s user_id=%(user_id)s path=%(path)s] "
    "%(message)s"
)


class RequestContextFormatter(std_logging.Formatter):
    DEFAULTS = {
        "request_id": "-",
        "session_id": "-",
        "user_id": "-",
        "path": "-",
    }

    def format(self, record: std_logging.LogRecord) -> str:
        ctx = get_request_context()
        for field, default in self.DEFAULTS.items():
            if not hasattr(record, field):
                setattr(record, field, ctx.get(field, default) or default)
        return super().format(record)


class RequestContextFilter(std_logging.Filter):
    def filter(self, record: std_logging.LogRecord) -> bool:
        ctx = get_request_context()
        record.request_id = ctx.get("request_id", "-")
        record.session_id = ctx.get("session_id", "-")
        record.user_id = ctx.get("user_id", "-")
        record.path = ctx.get("path", "-")
        return True


class RedactingFilter(std_logging.Filter):
    def filter(self, record: std_logging.LogRecord) -> bool:
        if settings.logging_redact_enabled:
            record.msg = redactor.redact_text(str(record.msg))
            if record.args:
                if isinstance(record.args, dict):
                    record.args = redactor.redact_value(record.args)
                elif isinstance(record.args, tuple):
                    record.args = tuple(redactor.redact_value(item) for item in record.args)
        return True


def setup_logging():
    level = builtins.getattr(std_logging, settings.log_level.upper(), std_logging.INFO)
    std_logging.basicConfig(level=level)
    root = std_logging.getLogger()
    root.setLevel(level)

    if not root.handlers:
        root.addHandler(std_logging.StreamHandler())

    for handler in root.handlers:
        handler.setFormatter(RequestContextFormatter(LOG_FORMAT))
        if not any(isinstance(item, RequestContextFilter) for item in handler.filters):
            handler.addFilter(RequestContextFilter())
        if not any(isinstance(item, RedactingFilter) for item in handler.filters):
            handler.addFilter(RedactingFilter())
