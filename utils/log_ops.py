import os
import logging
import datetime
from logging.handlers import TimedRotatingFileHandler

UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(UTILS_DIR)
LOG_DIR = os.path.join(ROOT_DIR, "logs")


def get_logger(module_name: str, log_path: str = None, interval: int = 7) -> logging.Logger:
    """지정된 모듈에 대한 로깅 기능을 설정하고 로거를 반환

    Args
        module_name (str): 모듈 이름.
        log_path (str): 로그 파일의 경로 및 이름 (optional).
                     파일 이름은 '.log'로 끝나야함
                     만약 지정하지 않으면, 로그는 LOG_DIR 내에 모듈 이름과 현재 날짜로 생성
        interval (int): 로그를 백업할 기간 (optional=7)
                     지정하지 않으면 7일마다 새로운 log파일을 생성
    Returns:
        (logging.Logger)

    Example:
        >>> logger = get_logger("my_module", "my_logs.log")
        >>> logger.info("이것은 정보 로그 메시지입니다.")
        >>> logger.error("이것은 오류 로그 메시지입니다.")

        >>> another_logger = get_logger("another_module")
        >>> another_logger.debug("이것은 디버그 로그 메시지입니다.")

    """

    if log_path and not log_path.endswith(".log"):
        msg = f"passed log_path ({log_path}), " "expected log_path must be ended with '.log'"
        raise ValueError(msg)

    elif not log_path:
        start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        log_module_dir = os.path.join(LOG_DIR, module_name)

        os.makedirs(log_module_dir, exist_ok=True)

        log_path = os.path.join(log_module_dir, f"{module_name}-{start_time}.log")
    else:
        log_dir = os.path.dirname(log_path)
        os.makedirs(log_dir, exist_ok=True)

    logger_formatter = logging.Formatter(
        fmt="{asctime}\t{name}\t{filename}:{lineno}\t{levelname}\t{message}",
        datefmt="%Y-%m-%dT%H:%M:%S",
        style="{",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logger_formatter)
    stream_handler.setLevel(logging.INFO)

    file_handler = TimedRotatingFileHandler(
        filename=log_path, when="D", interval=interval, backupCount=0
    )
    file_handler.setFormatter(logger_formatter)
    file_handler.setLevel(logging.DEBUG)

    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger
