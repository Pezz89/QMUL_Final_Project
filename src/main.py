#!/usr/bin/env python

import numpy as np
import argparse
import sys
import os
import loggerops

modpath = sys.argv[0]
modpath = os.path.splitext(modpath)[0]+'.log'

def extant_file(x):
    """
    'Type' for argparse - checks that directory exists.
    """
    if not os.path.isdir(x):
        # Argparse uses the ArgumentTypeError to give a rejection message like:
        # error: argument input: x does not exist
        raise argparse.ArgumentTypeError("{0} does not exist".format(x))
    return os.path.abspath(x)

def parse_arguments():
    """
    Parses arguments
    Returns a namespace with values for each argument
    """
    parser = argparse.ArgumentParser(
        description='Script for the classification of PCG data.',
    )

    # Get directory of test database
    parser.add_argument(
        dest="test_dir", type=extant_file,
        help="input file with two matrices", metavar="FILE"
    )

    parser.add_argument(
        '--verbose',
        '-v',
        action='count',
        help='Specifies level of verbosity in output. For example: \'-vvvvv\' '
        'will output all information. \'-v\' will output minimal information. '
    )
    args = parser.parse_args()
    if not args.verbose:
        args.verbose = 20
    else:
        levels = [50, 40, 30, 20, 10]
        if args.verbose > 5:
            args.verbose = 5
        args.verbose -= 1
        args.verbose = levels[args.verbose]
    return args

def main():
    # Process commandline arguments
    args = parse_arguments()
    logger = loggerops.create_logger(
        logger_streamlevel=args.verbose,
        log_filename=modpath,
        logger_filelevel=args.verbose
    )
    logger.debug("test")

if __name__ == "__main__":
    main()
