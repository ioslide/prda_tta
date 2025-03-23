import os
import sys
from loguru import logger as log
import time
import errno
__all__ = [
    "set_logger"
]

def set_logger(log_path, log_name):
    
    if log_path:
        try:
            log_path = f"{log_path}/" + time.strftime(
                "%Y-%m-%d", time.localtime()
            )
            os.makedirs(log_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    log_name = f"{log_name}_" + time.strftime("%H-%M-%S", time.localtime())
    if not os.path.isdir(log_path): os.makedirs()(log_path)
    log.add(
        os.path.join(log_path, f"{log_name}.txt"),
        rotation="50 MB",
        encoding="utf-8",
        enqueue=True,
        backtrace=True,
        diagnose=True,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level}</level> | <level>{file}</level> | <cyan>{name}</cyan>:<cyan>{module}.{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level> - <level>{exception}</level> - <level>{process}</level>",
    )