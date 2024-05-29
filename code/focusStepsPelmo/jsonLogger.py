from argparse import ArgumentParser, Namespace
import logging
import json


class JsonFormatter(logging.Formatter):
    """
    Formatter that outputs JSON strings after parsing the LogRecord.

    @param dict fmt_dict: Key: logging format attribute pairs. Defaults to {"message": "message"}.
    @param str time_format: time.strftime() format string. Default: "%Y-%m-%dT%H:%M:%S"
    @param str msec_format: Microsecond formatting. Appended at the end. Default: "%s.%03dZ"
    """
    def __init__(self, fmt_dict: dict = None, time_format: str = "%Y-%m-%dT%H:%M:%S", msec_format: str = "%s.%03dZ"):
        self.fmt_dict = fmt_dict if fmt_dict is not None else {"message": "message"}
        self.default_time_format = time_format
        self.default_msec_format = msec_format
        self.datefmt = None

    def usesTime(self) -> bool:
        """
        Overwritten to look for the attribute in the format dict values instead of the fmt string.
        """
        return "asctime" in self.fmt_dict.values()

    def formatMessage(self, record) -> dict:
        """
        Overwritten to return a dictionary of the relevant LogRecord attributes instead of a string. 
        KeyError is raised if an unknown attribute is provided in the fmt_dict. 
        """
        return {fmt_key: record.__dict__[fmt_val] for fmt_key, fmt_val in self.fmt_dict.items()}

    def format(self, record) -> str:
        """
        Mostly the same as the parent's class method, the difference being that a dict is manipulated and dumped as JSON
        instead of a string.
        """
        record.message = record.getMessage()
        
        if self.usesTime():
            record.asctime = self.formatTime(record, self.datefmt)

        message_dict = self.formatMessage(record)

        if record.exc_info:
            # Cache the traceback text to avoid converting it multiple times
            # (it's constant anyway)
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)

        if record.exc_text:
            message_dict["exc_info"] = record.exc_text

        if record.stack_info:
            message_dict["stack_info"] = self.formatStack(record.stack_info)

        return json.dumps(message_dict, default=str)
    
def get_json_handler(file: str, format: dict = {"module": "module", "time": "asctime", "level": "levelname", "thread": "threadName", "function": "funcName", "message": "message"}):
    file_handler = logging.FileHandler(file)
    json_formatter = JsonFormatter(format)
    file_handler.setFormatter(json_formatter)
    return file_handler

def add_log_args(parser: ArgumentParser):
    parser.add_argument('--loglevel', default=logging.INFO, choices=[logging.DEBUG, logging.INFO, logging.WARN, logging.ERROR, logging.CRITICAL], type=_str_to_level, help="The log level")
    parser.add_argument('--logfile', default='log.json', help="The logfile. Deaults to log.json")

def configure_logger_from_argparse(logger: logging.Logger, args: Namespace):
    logger.handlers = [get_json_handler(args.logfile)]
    logger.setLevel(args.loglevel)

def _str_to_level(level: str) -> int:
    d = {"debug": logging.DEBUG,
         "info": logging.INFO,
         "warning": logging.WARNING,
         "error": logging.ERROR,
         "critical": logging.CRITICAL}
    return d[level.lower()]