import logging


class Logger:
    """
    Custom logger class that can be used to get the same logger across the application.
    """

    _logger = {}

    @classmethod
    def get_logger(cls, name="root", level=logging.DEBUG, file_name=None, **kwargs):
        if not name in cls._logger:
            # create logger
            cls._logger[name] = logging.getLogger(name)
            cls._logger[name].setLevel(level)
            cls._logger[name].propagate = False

            # add 'levelname_c' attribute to log resords
            orig_record_factory = logging.getLogRecordFactory()
            log_colors = {
                logging.DEBUG: "\033[1;34m",  # blue
                logging.INFO: "\033[1;32m",  # green
                logging.WARNING: "\033[1;35m",  # magenta
                logging.ERROR: "\033[1;31m",  # red
                logging.CRITICAL: "\033[1;41m",  # red reverted
            }

            def record_factory(*args, **kwargs):
                record = orig_record_factory(*args, **kwargs)
                record.levelname_c = "{}{}{}".format(
                    log_colors[record.levelno], record.levelname, "\033[0m"
                )
                return record

            logging.setLogRecordFactory(record_factory)

            # create console handler and set level to debug
            ch = logging.StreamHandler()
            ch.setLevel(level)

            # create formatter
            formatter = logging.Formatter(
                "[%(asctime)s] - %(filename)s:%(funcName)s - %(levelname)s - %(message)s"
            )

            # add formatter to ch
            ch.setFormatter(formatter)

            # add ch to logger
            if cls._logger[name].hasHandlers():
                cls._logger[name].handlers.clear()
            cls._logger[name].addHandler(ch)

            if file_name:
                # create file handler and set level to debug
                fh = logging.FileHandler(file_name)
                fh.setLevel(level)
                fh.setFormatter(formatter)
                cls._logger[name].addHandler(fh)

        return cls._logger[name]
