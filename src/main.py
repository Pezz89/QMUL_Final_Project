#!/usr/bin/env python

import numpy as np
import argparse
import sys
import os
import glob
import loggerops
import logging
import pathops
from subprocess import Popen, PIPE
from generateFeatures import generateFeatures
from evaluateFeatures import evaluateFeatures
from buildClassifier import optimizeClassifierModel, scoreOptimizedModel, group_train_test_split, modelFeatureSelection
from resample import bootstrapResample, jacknifeResample, combinationResample, groupResample
from group import generateGroups
import pandas as pd
import traceback

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
        "--features-fname", "-o", type=str,
        default='features.pkl',
        help="Specify the name of the file to save generated features to for "
        "future use", metavar="OUTFNAME"
    )

    parser.add_argument(
        "--segment",
        action="store_true",
        help="Run Matlab segmentation script to create segmentation analysis"
    )

    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run optimization algorithm to find best model and parameters "
        "for classifier"
    )

    parser.add_argument(
        '--eval',
        '-e',
        type=int,
        default=50,
        help='Number of evaluation to pass to the particle swarm optimization'
    )

    parser.add_argument(
        "--select-features",
        type=int,
        default=50,
        help="Run feature selection algorithm to find best features for model, "
        "either selecting or reducing features by the integer specified. This "
        "depends on use of --backward flag, to determine forward or backward "
        "feature selection. (a value of 0 skips feature selection entirely, "
        "using previously generated features if available. A value less than 0 "
        "uses all available features.)"
    )

    parser.add_argument(
        "--backward",
        "-b",
        action="store_true",
        help="Runs backward feature selection as opposed to default forward selection."
    )

    parser.add_argument(
        "--parameters_fname", type=str,
        default='parameters.h5',
        help="Specify the name of the file to save generated features to for "
        "future use", metavar="OUTFNAME"
    )

    parser.add_argument(
        "--fs_fname", type=str,
        default='feature_selection.h5',
        help="Specify the name of the file to save generated feature selection model to for "
        "future use", metavar="OUTFNAME"
    )

    parser.add_argument(
        "--no-parallel",
        "-p",
        action="store_true",
        help="Disable processing in parallel. (Will likely decrease performance "
        "but may aid in debugging)"
    )

    parser.add_argument(
        "--reanalyse",
        action="store_true",
        help="Force regeneration of database features"
    )

    parser.add_argument(
        '--verbose',
        '-v',
        action='count',
        help='Specifies level of verbosity in output. For example: \'-vvvvv\' '
        'will output all information. \'-v\' will output minimal information. '
    )

    parser.add_argument(
        '--resample-mix',
        '-r',
        type=float,
        default=0.0,
        help='Mix between bootstrap and jacknife resampling used to balance '
        'the dataset (0=just jacknife, 1=just bootsrap)'
    )

    parser.add_argument(
        '--keep-logs',
        action="store_true",
        help="Remove previously generated logs"
    )




    args = parser.parse_args()
    # Set verbosity of logger output based on argument
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
    if not args.keep_logs:
        for log in glob.glob("./*.log"):
            os.remove(log)

    global logger
    logger = loggerops.create_logger(
        logger_streamlevel=args.verbose,
        log_filename=modpath,
        logger_filelevel=args.verbose
    )

    # Run PCG segmentation using algorithm provided by the Physionet Challenge
    # 2016, using MATLAB
    if args.segment:
        logger.info("Running MATLAB segmentation...")
        runSpringerSegmentation(args.test_dir, args.output_dir)
    # Parallel processing defaults to on
    parallelize = not args.no_parallel

    # Search recursively through folders to find PCG data and supporting files
    # (class labels, noise labels)
    dataFilepaths = getFilepaths(args.test_dir, args.output_dir)
    # Calculate all features for dataset
    features = generateFeatures(dataFilepaths, args.output_dir, args.features_fname, parallelize=parallelize, reanalyse=args.reanalyse)

    # Get classification labels for all loaded files, using filenames stored in
    # the 'features' DataFrame
    classifications = getClassifications(args.test_dir, features)

    # Resample generated features and classifications to balance dataset, using
    # a combination of jacknife and bootstrap resampling based on user input
    features, classifications = groupResample(features, classifications, mix=args.resample_mix)
    # Filter any eroneous values from features
    # -inf occurs as a result of log(0)
    # TODO: fix these value in feature generation on per-feature basis
    features = features.replace(-np.inf, 0)
    features = features.replace(np.inf, np.nan)

    # Get filepath for storing model parameters generated in optimization
    parameters_filepath = os.path.join(args.output_dir, args.parameters_fname)
    fs_filepath = os.path.join(args.output_dir, args.fs_fname)

    # Group samples by sub-database, for use with Leave-one-out cross
    # validation
    groups = generateGroups(features)

    # Split features into training and test set by database
    train_features, test_features, train_classifications, test_classifications, train_groups, test_groups = group_train_test_split(features, classifications, groups)

    if args.optimize:
        # Optimize model parameters using particle swarm optimization
        optimizeClassifierModel(train_features, train_classifications, train_groups, parameters_filepath, num_evals=args.eval, parallelize=parallelize)

    if args.select_features > 0:
        # Run sequential feature selection to select optimal set of features
        # for training
        modelFeatureSelection(train_features, train_classifications, parameters_filepath, fs_filepath, feature_count=args.select_features, backward=args.backward)

    # Score model using optimized parameters and features
    scoreOptimizedModel(features, classifications, groups, train_features, test_features, train_classifications, test_classifications, parameters_filepath, feature_selection=args.select_features)


'''
Read classification labels for files in the dataset
'''
def getClassifications(referenceLocation, features):
    # Find all reference files
    refLocs = []
    for root, dirs, files in os.walk(referenceLocation):
        for file in files:
            if file == 'REFERENCE-SQI.csv':
                refLocs.append(os.path.join(root, file))
    refLocs = sorted(refLocs)
    classifications = pd.DataFrame({'class': [], 'quality': []})
    for refFile in refLocs:
        refDF = pd.read_csv(refFile,header=None, index_col=0, names=['class', 'quality'])
        classifications = classifications.append(refDF)
    classifications = classifications.ix[features.index]

    return classifications


'''
Search folder for audio and segment files to be used as samples
'''
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

'''
Run external segmentation algorithm on PCG files
'''
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
        '-nojvm',
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
