"""Create a logger for applications to handle stream and file logging."""

import logging
import types
import pdb

def log_newline(self, how_many_lines=1):
    # Switch handler, output a blank line
    self.removeHandler(self.console_handler)
    self.addHandler(self.blank_handler)
    for i in range(how_many_lines):
        self.info('')

    # Switch back
    self.removeHandler(self.blank_handler)
    self.addHandler(self.console_handler)

def create_logger(
    name=None,
    logger_streamlevel=logging.DEBUG,
    log_filename=None,
    logger_filelevel=logging.DEBUG,
    use_stream_handler=True
):
    """
    Create a main program logger with generic config.

    Example use:
        # Creates a logger with the name of the script running and configures
        # to default values for standard use cases.
        logger = create_logger(__name__)
    """
    # Create a logger object to use in the main script of an application
    logger = logging.getLogger(name=name)
    # Set the level of messages to be handled by the logger. defaults to all
    # messages
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)8s] --- "
                                  "%(message)s (%(filename)s:%(lineno)s)",
                                  "%Y-%m-%d %H:%M:%S")

    # If a log file path has been provided...
    if log_filename:
        # Create file handler to save logging messages to.
        file_handler = logging.FileHandler(log_filename)
        # Set the level of messages to save to the log file
        file_handler.setLevel(logging.DEBUG)

        # Create a formatter to set message style.
        # Add the formatter to the file handler.
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    if use_stream_handler:
        # Create a stream handler to send messages to the output.
        stream_handler = logging.StreamHandler()
        # Add formatter to the stream handler
        stream_handler.setFormatter(formatter)
        # Set the level of messages to stream to the output
        stream_handler.setLevel(logger_streamlevel)

        # Add the handlers to the logger
        logger.addHandler(stream_handler)
    else:
        logger.removeHandler(stream_handler)


    return logger


def set_stream_level(logger, log_level=logging.DEBUG):
    """Change all stream handler to use given log_level."""
    logger.setLevel(log_level)
    for hh in logger.handlers:
        if hasattr(hh, 'stream'):
            hh.setLevel(log_level)
