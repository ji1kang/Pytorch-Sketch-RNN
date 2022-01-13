import logging


class Logger:
    def __init__(self, logger_name, log_fname=None, level=logging.INFO):
        import os

        logger = logging.getLogger(logger_name)
        logger.setLevel(level)  # DEBUG < INFO < WARNING < ERROR < CRITICAL
        formatter = logging.Formatter(
            '[%(asctime)s][%(levelname)s|%(filename)s:%(lineno)s] >> %(message)s')
        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)
        logger.addHandler(streamHandler)

        if log_fname is not None:
            if not os.path.exists('./log'):
                os.makedirs('./log')

            fileHandler = logging.FileHandler(log_fname)
            fileHandler.setFormatter(formatter)
            logger.addHandler(fileHandler)

        self._logger = logger

    @property
    def logger(self):
        return self._logger
