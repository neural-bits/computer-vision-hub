import logging
import sys
from time import sleep

import coloredlogs


def setup_logger(logger_name, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] [%(funcName)s] [%(lineno)d] - %(message)s"
    )
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(streamHandler)


def logger_configure():
    root = logging.getLogger()
    coloredlogs.install(
        level=logging.DEBUG,
        logger=root,
        fmt="[%(asctime)s] [%(name)s] [%(levelname)s] [%(processName)s->%(module)s] [%(lineno)-3d] - %(message)s",
        level_styles=dict(
            debug=dict(color="white"),
            info=dict(color="blue"),
            warning=dict(color="yellow", bright=True),
            error=dict(color="red", bold=True, bright=True),
            critical=dict(color="black", bold=True, background="red"),
        ),
        field_styles=dict(
            asctime=dict(color="yellow"),
            name=dict(color="white"),
            processName=dict(color="green"),
            funcName=dict(color="green"),
            lineno=dict(color="blue"),
            message=dict(color="white"),
        ),
    )
    return root


def listener_process(queue):
    logger_configure()
    while True:
        while not queue.empty():
            record = queue.get()
            logger = logging.getLogger(record.name)
            logger.handle(record)
        sleep(1)
