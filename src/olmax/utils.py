import logging
import os
import sys
from datetime import datetime
from typing import Any, Type

import rich
from rich.console import Console, ConsoleRenderable
from rich.highlighter import NullHighlighter
from rich.text import Text
from rich.traceback import Traceback

from .exceptions import OLMaXError

log = logging.getLogger(__name__)

_LOGGING_CONFIGURED = False


def setup_logging(force: bool = False) -> None:
    """
    Configure logging.

    :param force: Force configuring logging even if it was already configured.
    """
    global _LOGGING_CONFIGURED

    if _LOGGING_CONFIGURED and not force:
        return

    handler: logging.Handler
    if os.environ.get("BEAKER_EXPERIMENT_ID") is None and (
        os.environ.get("DEBIAN_FRONTEND", None) == "noninteractive" or not sys.stdout.isatty()
    ):
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s\t%(name)s:%(lineno)s\t%(levelname)s\t%(message)s"
        )
        formatter.default_time_format = "%Y-%m-%d %H:%M:%S"
        formatter.default_msec_format = "%s.%03d"
        handler.setFormatter(formatter)
    else:
        handler = _RichHandler()

    logging.basicConfig(handlers=[handler], level=logging.INFO, force=True)
    logging.captureWarnings(True)
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    logging.getLogger("google").setLevel(logging.WARNING)

    _LOGGING_CONFIGURED = True


def logging_configured() -> bool:
    """
    Returns ``True`` if logging has been configured (like with :func:`setup_logging()`),
    otherwise returns ``False``.
    """
    if _LOGGING_CONFIGURED:
        return True
    else:
        # Otherwise check if the root logger has any handlers.
        return len(logging.getLogger().handlers) > 0


def _print_stderr(*args, **kwargs):
    Console(stderr=True).print(*args, **kwargs)


def _excepthook(exctype, value, tb):
    """
    Used to patch ``sys.excepthook`` in order to customize handling of uncaught exceptions.
    """
    in_house_error_types: list[Type[Exception]] = [OLMaXError]
    try:
        from gantry.exceptions import GantryError

        in_house_error_types.append(GantryError)
    except ImportError:
        pass

    try:
        from beaker.exceptions import BeakerError

        in_house_error_types.append(BeakerError)
    except ImportError:
        pass

    # Ignore in-house error types because we don't need a traceback for those.
    if issubclass(exctype, tuple(in_house_error_types)):
        _print_stderr(f"[red][bold]{exctype.__name__}:[/] [i]{value}[/][/]")
    # For interruptions, call the original exception handler.
    elif issubclass(exctype, KeyboardInterrupt):
        sys.__excepthook__(exctype, value, tb)
    else:
        _print_stderr(Traceback.from_exception(exctype, value, tb))


def install_excepthook():
    sys.excepthook = _excepthook


def prepare_cli_environment():
    install_excepthook()
    setup_logging()


def set_env_var(name: str, value: str, override: bool = False, secret: bool = False):
    value_str = "****" if secret else value
    if name in os.environ:
        if override and os.environ[name] != value:
            msg = f"Overriding env var '{name}' to '{value_str}'"
            if logging_configured():
                log.warning(msg)
            else:
                print(msg)
            os.environ[name] = value
    else:
        msg = f"Setting env var '{name}' to '{value_str}'"
        if logging_configured():
            log.info(msg)
        else:
            print(msg)
        os.environ[name] = value


def mib_to_bytes(mb: float) -> int:
    return int(1024 * 1024 * mb)


def bytes_to_mib(b: int) -> float:
    return b / (1024 * 1024)


class _RichHandler(logging.Handler):
    """
    A simplified version of rich.logging.RichHandler from
    https://github.com/Textualize/rich/blob/master/rich/logging.py
    """

    def __init__(
        self,
        *,
        level: int | str = logging.NOTSET,
        console: Console | None = None,
        markup: bool = False,
    ) -> None:
        super().__init__(level=level)
        self.console = console or rich.get_console()
        self.highlighter = NullHighlighter()
        self.markup = markup

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if hasattr(record.msg, "__rich__") or hasattr(record.msg, "__rich_console__"):
                self.console.print(record.msg)
            else:
                msg: Any = record.msg
                if isinstance(record.msg, str):
                    msg = self.render_message(record=record, message=record.getMessage())
                renderables = [
                    self.get_time_text(record),
                    self.get_level_text(record),
                    self.get_location_text(record),
                    msg,
                ]
                if record.exc_info is not None:
                    tb = Traceback.from_exception(*record.exc_info)  # type: ignore
                    renderables.append(tb)
                self.console.print(*renderables)
        except Exception:
            self.handleError(record)

    def render_message(self, *, record: logging.LogRecord, message: str) -> ConsoleRenderable:
        use_markup = getattr(record, "markup", self.markup)
        message_text = Text.from_markup(message) if use_markup else Text(message)

        highlighter = getattr(record, "highlighter", self.highlighter)
        if highlighter:
            message_text = highlighter(message_text)

        return message_text

    def get_time_text(self, record: logging.LogRecord) -> Text:
        log_time = datetime.fromtimestamp(record.created)
        time_str = log_time.strftime("[%Y-%m-%d %X]")
        return Text(time_str, style="log.time", end=" ")

    def get_level_text(self, record: logging.LogRecord) -> Text:
        level_name = record.levelname
        level_text = Text.styled(level_name.ljust(8), f"logging.level.{level_name.lower()}")
        level_text.style = "log.level"
        level_text.end = " "
        return level_text

    def get_location_text(self, record: logging.LogRecord) -> Text:
        name_and_line = f"{record.name}:{record.lineno}" if record.name != "root" else "root"
        text = f"[{name_and_line}]"  # type: ignore
        return Text(text, style="log.path")
