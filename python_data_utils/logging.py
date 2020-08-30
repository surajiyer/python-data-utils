import logging
import os


# initialize logging
log_configs = {
    'level': logging.INFO,
    'format': '%(asctime)s %(filename)s:%(lineno)d %(levelname)s  %(message)s',
    'datefmt': '%Y-%m-%d %X'
}
logging.basicConfig(**log_configs)
__log_files = []


def get_logger(path=None, file=None):
    global __log_files
    logger = logging.getLogger()

    # create log file
    if path and file:
        log_file = os.path.join(path, file)

        if log_file not in __log_files:
            # create the log file (if required)
            if not os.path.isfile(log_file):
                open(log_file, "w+").close()

            # create file handler
            handler = logging.FileHandler(log_file)
            handler.setLevel(log_configs['level'])
            formatter = logging.Formatter(
                log_configs['format'],
                datefmt=log_configs['datefmt'])
            handler.setFormatter(formatter)
            logger.addHandler(handler)

            __log_files.append(log_file)

    return logger
