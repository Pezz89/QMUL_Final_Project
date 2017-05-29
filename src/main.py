#!/usr/bin/env python

import numpy as np
import argparse
import sys
import os
import loggerops
import logging
from subprocess import Popen, PIPE

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
        help="Directory of test data to train the system", metavar="TESTDIR"
    )
    parser.add_argument(
        dest="output_dir", type=extant_file,
        help="Directory to store output analyses", metavar="OUTDIR"
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

    global logger
    logger = loggerops.create_logger(
        logger_streamlevel=args.verbose,
        log_filename=modpath,
        logger_filelevel=args.verbose
    )

    runSpringerSegmentation(args.test_dir, args.output_dir)

def runSpringerSegmentation(dataset_dir, output_dir):
    cmd = ['matlab', '-nosplash', '-nodesktop', '-r', 'try addpath(\'./SpringerExtraction\'); main(\'{0}\', \'{1}\'); catch err; disp(err); disp(err.message); disp(err.stack); end; quit'.format(dataset_dir, output_dir)]

    logger.debug("Running external matlab command:\n" + ' '.join(cmd))

    process = Popen(cmd, stdout=sys.stdout, stderr=sys.stderr).wait()
    stdout, stderr = process.communicate()

if __name__ == "__main__":
    main()
