import logging

_DEF_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"

_def_level = logging.INFO


def setup_logging(level: int = _def_level) -> logging.Logger:
    logging.basicConfig(level=level, format=_DEF_FORMAT)
    return logging.getLogger("voxtral")
