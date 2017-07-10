#!/usr/bin/env python

import numpy as np
import argparse
import sys
import os
import loggerops
import logging
import pathops
from subprocess import Popen, PIPE
from generateFeatures import generateFeatures, normaliseFeatures
from evaluateFeatures import evaluateFeatures
from buildClassifier import optimizeClassifierModel, scoreOptimizedModel, group_train_test_split
from resample import bootstrapResample, jacknifeResample, combinationResample, groupResample
from group import generateGroups
import pandas as pd
import traceback
import StringIO

import pdb

modpath = sys.argv[0]
modpath = os.path.splitext(modpath)[0]+'.log'

def add_custom_print_exception():
    old_print_exception = traceback.print_exception
    def custom_print_exception(etype, value, tb, limit=None, file=None):
        tb_output = StringIO.StringIO()
        traceback.print_tb(tb, limit, tb_output)
        logger = logging.getLogger(__name__)
        logger.error(tb_output.getvalue())
        tb_output.close()
        old_print_exception(etype, value, tb, limit=None, file=None)
    traceback.print_exception = custom_print_exception

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
        "--features_fname", "-o", type=str,
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
        "--parameters_fname", type=str,
        default='parameters.pkl',
        help="Specify the name of the file to save generated features to for "
        "future use", metavar="OUTFNAME"
    )

    parser.add_argument(
        "--no_parallel",
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
        '--resample_mix',
        '-r',
        type=float,
        default=0.5,
        help='Mix between bootstrap and jacknife resampling used to balance '
        'the dataset (0=just jacknife, 1=just bootsrap)'
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

    # This code was taken from: https://stackoverflow.com/questions/19425736/how-to-redirect-stdout-and-stderr-to-logger-in-py
    class LoggerWriter:
        def __init__(self, level):
            # self.level is really like using log.debug(message)
            # at least in my case
            self.level = level

        def write(self, message):
            # if statement reduces the amount of newlines that are
            # printed to the logger
            mess = [s for s in message.splitlines()]
            for m in mess:
                self.level(m.ljust(92))

        def flush(self):
            # create a flush method so things can be flushed when
            # the system wants to. Not sure if simply 'printing'
            # sys.stderr is the correct way to do it, but it seemed
            # to work properly for me.
            pass
            #self.level(sys.stderr)

    global logger
    logger = loggerops.create_logger(
        logger_streamlevel=args.verbose,
        log_filename=modpath,
        logger_filelevel=args.verbose
    )
    #sys.stdout = LoggerWriter(logger.info)
    #sys.stderr = LoggerWriter(logger.debug)

    #add_custom_print_exception()
    if args.segment:
        logger.info("Running MATLAB segmentation...")
        runSpringerSegmentation(args.test_dir, args.output_dir)
    parallelize = not args.no_parallel
    dataFilepaths = getFilepaths(args.test_dir, args.output_dir)
    features = generateFeatures(dataFilepaths, args.output_dir, args.features_fname, parallelize=parallelize, reanalyse=args.reanalyse)
    features = normaliseFeatures(features)
    classifications = getClassifications(args.test_dir, features)
    features, classifications = groupResample(features, classifications, mix=args.resample_mix)
    evaluateFeatures(features, classifications)

    parameters_filepath = os.path.join(args.output_dir, args.parameters_fname)
    # Split features into training and test set by database
    groups = generateGroups(features)
    train_features, test_features, train_classifications, test_classifications, train_groups, test_groups = group_train_test_split(features, classifications, groups)

    train_features, test_features = apply_pca(train_features, test_features, train_classifications, test_classifications)
    if args.optimize:
        optimizeClassifierModel(train_features, train_classifications, train_groups, parameters_filepath, parallelize=parallelize)
    scoreOptimizedModel(train_features, test_features, train_classifications, test_classifications, parameters_filepath)

def apply_pca(train_X, test_X, train_Y, test_Y):
    from sklearn.decomposition import KernelPCA

    train_rows = train_X.index
    test_rows = test_X.index

    kpca = KernelPCA()
    kpca.fit(train_X)

    X_pca = kpca.transform(train_X)
    import matplotlib.pyplot as plt



    tot = sum(kpca.lambdas_)
    var_exp = [(i / tot)*100 for i in sorted(kpca.lambdas_, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)

    if False:
        with plt.style.context('seaborn-whitegrid'):
            fig, ax = plt.subplots(figsize=(6, 4))
            plt.bar(range(X_pca.shape[1]), var_exp, alpha=0.5, align='center',
                    label='individual explained variance')
            plt.step(range(X_pca.shape[1]), cum_var_exp, where='mid',
                    label='cumulative explained variance')
            plt.ylabel('Explained variance ratio')
            plt.xlabel('Principal components')
            plt.xticks(range(X_pca.shape[1]))
            ax.set_xticklabels(np.arange(1, train_X.shape[1] + 1))
            plt.legend(loc='best')
            plt.tight_layout()
            plt.show()

    train_X = X_pca[:, :30]

    X_pca = kpca.transform(test_X)

    tot = sum(kpca.lambdas_)
    var_exp = [(i / tot)*100 for i in sorted(kpca.lambdas_, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)

    if False:
        with plt.style.context('seaborn-whitegrid'):
            fig, ax = plt.subplots(figsize=(6, 4))
            plt.bar(range(X_pca.shape[1]), var_exp, alpha=0.5, align='center',
                    label='individual explained variance')
            plt.step(range(X_pca.shape[1]), cum_var_exp, where='mid',
                    label='cumulative explained variance')
            plt.ylabel('Explained variance ratio')
            plt.xlabel('Principal components')
            plt.xticks(range(X_pca.shape[1]))
            ax.set_xticklabels(np.arange(1, X.shape[1] + 1))
            plt.legend(loc='best')
            plt.tight_layout()
            plt.show()


    test_X = X_pca[:, :30]
    return pd.DataFrame(train_X, index=train_rows), pd.DataFrame(test_X, index=test_rows)

'''
Read classification labels for files in the dataset
'''
def getClassifications(referenceLocation, features):
    # Find all reference files
    refLocs = []
    for root, dirs, files in os.walk(referenceLocation):
        for file in files:
            if file.endswith('.csv'):
                refLocs.append(os.path.join(root, file))
    refLocs = sorted(refLocs)
    classifications = pd.Series([])
    for refFile in refLocs:
        classifications = classifications.append(pd.Series.from_csv(refFile))
    classifications = classifications[features.index]
    return classifications


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
