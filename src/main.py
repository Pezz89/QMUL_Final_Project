#!/usr/bin/env python

import numpy as np
import argparse
import sys
import os
import loggerops
import logging
import pathops
from subprocess import Popen, PIPE
from generateFeatures import generateFeatures

import pdb

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

    # Get directory for storing output analyses
    parser.add_argument(
        dest="output_dir", type=extant_file,
        help="Directory to store output analyses", metavar="OUTDIR"
    )

    parser.add_argument(
        "--segment",
        action="store_true",
        help="Run Matlab segmentation script to create segmentation analysis"
    )

    parser.add_argument(
        "--parallelize",
        "-p",
        action="store_true",
        help="Run processing in parallel where possible to decrease increase"
        " performance"
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

    if args.segment:
        logger.info("Running MATLAB segmentation...")
        runSpringerSegmentation(args.test_dir, args.output_dir)
    dataFilepaths = getFilepaths(args.test_dir, args.output_dir)
    generateFeatures(dataFilepaths, args.output_dir, parallelize=args.parallelize)

def getFilepaths(audioLocation, segmentsLocation):
    # Find all segmentation files
    segLocs = []
    for root, dirs, files in os.walk(segmentsLocation):
        for file in files:
            if file.endswith('.csv'):
                segLocs.append(os.path.join(root, file))
    segLocs = sorted(segLocs)
    # Find all source PCG audio files
    audioLocs = []
    for root, dirs, files in os.walk(audioLocation):
        for file in files:
            if file.endswith('.wav'):
                audioLocs.append(os.path.join(root, file))
    audioLocs = sorted(audioLocs)

    # Create a list of tuples with filepaths for audio files and segmentation
    # files
    filepaths = []
    for segLoc in segLocs:
        segName = os.path.splitext(os.path.basename(segLoc))[0][:-5]
        for ind, audioLoc in enumerate(audioLocs):
            if os.path.splitext(os.path.basename(audioLoc))[0] == segName:
                filepaths.append({'name': segName, 'audio':audioLoc, 'seg':segLoc})
                del audioLocs[ind]
                break
    return filepaths

def runSpringerSegmentation(dataset_dir, output_root_dir):
    # Run matlab springer segmentation code to generate S1, S2, Systole and
    # Diastole segmentations
    output_dir = os.path.join(output_root_dir, "seg")
    pathops.dir_must_exist(output_dir)
    # Build command for calling the matlab segmentation script to output
    # segments from the input dataset to a seg subdirectory of the output
    # directory.
    cmd = [
        'matlab',
        '-nosplash',
        '-nodesktop',
        '-r',
        'try addpath(\'./SpringerExtraction\');'
        'main(\'{0}\', \'{1}\');'
        'catch err;'
        'disp(err);'
        'disp(err.message);'
        'disp(err.stack);'
        'end;'
        'quit'.format(dataset_dir, output_dir)
    ]

    logger.debug("Running external matlab command:\n" + ' '.join(cmd))

    Popen(cmd, stdout=sys.stdout, stderr=sys.stderr).wait()
    # Hack to make sure matlab instances are killed at end of script...
    cmd = ["pkill", "MATLAB"]
    Popen(cmd, stdout=sys.stdout, stderr=sys.stderr).wait()

if __name__ == "__main__":
    main()
