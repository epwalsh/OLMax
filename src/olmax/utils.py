import logging
import os

log = logging.getLogger(__name__)

_LOGGING_CONFIGURED = False


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
