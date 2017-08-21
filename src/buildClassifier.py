# -*- coding: utf-8 -*-
from __future__ import division

################################################################################
# Generic imports
################################################################################
import re
import numpy as np
import pandas as pd
import pdb
import logging
import sys
import multiprocessing
import six
import textwrap
from tabulate import tabulate

################################################################################
# Scikit-Learn imports
################################################################################
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold, StratifiedShuffleSplit, LeaveOneGroupOut
from sklearn.metrics.scorer import make_scorer
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline

# Potential classifier models

from sklearn.tree import DecisionTreeClassifier
# k-nearest neighbours
from sklearn.neighbors import KNeighborsClassifier
# support vector machine classifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.linear_model import LogisticRegression


################################################################################
# MLXtend extensions to Scikit-Learn
################################################################################
from mlxtend.classifier import StackingCVClassifier

from mlxtend.feature_selection import SequentialFeatureSelector as SFS

################################################################################
# Particle swarm optimization library
################################################################################
import optunity
import optunity.search_spaces as search_spaces
import optunity.constraints as constraints


################################################################################
# Project imports
################################################################################
from physionetscore import score, sensitivity, specificity
from group import generateGroups
import loggerops
from multiscorer import MultiScorer


logger = logging.getLogger(__name__)
# Generate random seeds to ensure reproducible performance
random_state = np.random.RandomState(42)
np.random.seed(42)

'''
Parse arguments from particle swarm algorithm and generate stacking classifier
model.
'''
def buildModel(
    worker_log = logging.getLogger(__name__),
    **kwargs
):
    # Filter out None arguments from key word arguments
    kwargs = {k:v for k,v in kwargs.iteritems() if v is not None}

    # Separate arguments for each sub-classifier
    cl1_parameters = {k: v for k, v in kwargs.iteritems() if 'cl1_' in k}
    cl2_parameters = {k: v for k, v in kwargs.iteritems() if 'cl2_' in k}
    cl3_parameters = {k: v for k, v in kwargs.iteritems() if 'cl3_' in k}
    lr_parameters = {k: v for k, v in kwargs.iteritems() if 'lr_' in k}

    # A list of parameters that must be integer values
    interger_params = ['n_neighbors', 'max_features', 'max_depth', 'n_estimators', 'penalty']

    # Remove prefix used in search space for separating arguments with the same
    # names
    cl1_parameters = {k.replace('cl1_', ''): v.replace('cl1_', '') if isinstance(v, six.string_types) else v for k, v in cl1_parameters.iteritems() if 'cl1_' in k}
    cl2_parameters = {k.replace('cl2_', ''): v.replace('cl2_', '') if isinstance(v, six.string_types) else v for k, v in cl2_parameters.iteritems() if 'cl2_' in k}
    cl3_parameters = {k.replace('cl3_', ''): v.replace('cl3_', '') if isinstance(v, six.string_types) else v for k, v in cl3_parameters.iteritems() if 'cl3_' in k}
    lr_parameters = {k.replace('lr_', ''): v.replace('lr_', '') if isinstance(v, six.string_types) else v for k, v in lr_parameters.iteritems() if 'lr_' in k}

    # Get the name of the algorithm for each sub-classifier
    cl1_algorithm = cl1_parameters.pop('algorithm')
    cl2_algorithm = cl2_parameters.pop('algorithm')
    cl3_algorithm = cl3_parameters.pop('algorithm')

    # Round any float parameters that should be ints
    for p in set(interger_params).intersection(cl1_parameters):
        cl1_parameters[p] = int(round(cl1_parameters[p]))
    for p in set(interger_params).intersection(cl2_parameters):
        cl2_parameters[p] = int(round(cl2_parameters[p]))
    for p in set(interger_params).intersection(cl3_parameters):
        cl3_parameters[p] = int(round(cl3_parameters[p]))
    for p in set(interger_params).intersection(lr_parameters):
        lr_parameters[p] = int(round(lr_parameters[p]))

    # Map integers to string arguments for meta-classifier penalty argument
    penalty_str = ['l1', 'l2']
    lr_parameters['penalty'] = penalty_str[lr_parameters['penalty']]


    worker_log.info("--------------------------------------------------------------------------------------------")
    # Switch for selecting model objects for each sub-classifier
    model_methods = {
        'k-nn': KNeighborsClassifier,
        'SVM': SVC,
        'naive-bayes': GaussianNB,
        'random-forest': RandomForestClassifier,
        'adaboost': AdaBoostClassifier,
        'logistic-regression': LogisticRegression
    }

    worker_log.debug("Building Stacking Classifier, with parameters:".ljust(92))
    worker_log.debug(("Algorithm: {} ".format(cl1_algorithm) + " ".join(['{0}: {1}'.format(k, v) for k, v in cl1_parameters.iteritems()])).ljust(92))
    worker_log.debug(("Algorithm: {} ".format(cl2_algorithm) + " ".join(['{0}: {1}'.format(k, v) for k, v in cl2_parameters.iteritems()])).ljust(92))
    worker_log.debug(("Algorithm: {} ".format(cl3_algorithm) + " ".join(['{0}: {1}'.format(k, v) for k, v in cl3_parameters.iteritems()])).ljust(92))
    worker_log.debug("Meta-classifier (Logistic regression) parameters:".ljust(92))
    worker_log.debug(" ".join(['{0}: {1}'.format(k, v) for k, v in lr_parameters.iteritems()]).ljust(92))


    # Handle static arguments for individual classifier models
    if cl1_algorithm == 'adaboost':
        cl1_parameters['base_estimator'] = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
    if cl2_algorithm == 'adaboost':
        cl2_parameters['base_estimator'] = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
    if cl3_algorithm == 'adaboost':
        cl3_parameters['base_estimator'] = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
    if cl1_algorithm == 'SVM':
        cl1_parameters['probability'] = True
    if cl2_algorithm == 'SVM':
        cl2_parameters['probability'] = True
    if cl3_algorithm == 'SVM':
        cl3_parameters['probability'] = True

    # Build sub-classifier and meta-classifier models
    clf1 = model_methods[cl1_algorithm](**cl1_parameters)
    clf2 = model_methods[cl2_algorithm](**cl2_parameters)
    clf3 = model_methods[cl3_algorithm](**cl3_parameters)
    lr = LogisticRegression(random_state=42, **lr_parameters)

    # Create sklearn pipe using an imputer to handle Nan values, a scaler for
    # ensuring all values are in the range of 0-1 and the final stacking
    # classifier
    pipe_components = [("imputer", preprocessing.Imputer()), ("scaler", preprocessing.MinMaxScaler()), ("model", StackingCVClassifier(classifiers=[clf1, clf2, clf3], meta_classifier=lr, use_probas=True))]
    pipe = Pipeline(pipe_components)

    return pipe


def modelFeatureSelection(features, classifications, optimization_fpath, sfs_fpath, feature_count=50, worker_log=logging.getLogger(__name__), backward=False, **kwargs):
    # load model information from file
    with pd.HDFStore(optimization_fpath) as hdf:
        iterations = [extract_number(x)[0] for x in hdf.keys()]
        try:
            latestIteration = max(iterations)
        except ValueError:
            raise ValueError("Models have not been optimized, please run with the --optimize flag")
        latestSolution = hdf["/solution{}".format(latestIteration)]
    latestSolution = latestSolution.dropna()

    # Rebuild model from saved parameters
    # TODO: Replace rebuilding with pickling of original model builds, so that
    # they can be directly loaded back from file
    model = buildModel(**latestSolution)
    worker_log.info("--------------------------------------------------------------------------------------------")
    worker_log.info("Running feature selection...".ljust(92))
    worker_log.info("--------------------------------------------------------------------------------------------")

    # Create a scorer object using the custom Physionet challenge scoring
    # metrics
    physionetScorer = MultiScorer({
        'score' : (score, {'custom_y': classifications}),
        'sensitivity' : (sensitivity, {'custom_y': classifications}),
        'specificity' : (specificity, {'custom_y': classifications})
    })

    classifications = filterNoiseSamples(classifications)

    if backward:
        k_features=(features.shape[1]-feature_count, features.shape[1])
    else:
        k_features=(1, feature_count)

    # Wrap generated model in a sequential feature selection algorithm, for
    # dynamic feature reduction
    model = SFS(
        model,
        k_features=k_features,
        forward=~backward,
        floating=True,
        verbose=2,
        scoring=physionetScorer,
        cv=3,#gkf,
        logger=worker_log,
        n_jobs=-1
    )
    model.fit(features, classifications)

    # Save feature selection wrapped model for generation of plots
    joblib.dump(model, sfs_fpath)

    # Save best features to file
    bestLabel = features.columns[np.array(model.k_feature_idx_)]
    bestLabel.to_series().to_hdf(optimization_fpath, key="bestFeatures")


'''
Score intermediate models generated using particle swarm algorithm
Uses stratified 10-fold cross validation to calculate model performance
'''
def scoreModel(features, classifications, gkf, model, worker_log=logging.getLogger(__name__), **kwargs):
    # Create a scorer object using the custom Physionet challenge scoring
    # metrics
    physionetScorer = MultiScorer({
        'score' : (score, {'custom_y': classifications}),
        'sensitivity' : (sensitivity, {'custom_y': classifications}),
        'specificity' : (specificity, {'custom_y': classifications})
    })
    classifications = filterNoiseSamples(classifications)

    # Set precision of floats printed to logger
    np.set_printoptions(precision=4)

    worker_log.info("--------------------------------------------------------------------------------------------")
    # Run cross-validation using model and generate score
    skf = StratifiedKFold(n_splits=10, random_state=42)
    logo_scores = cross_val_score(model, features, classifications, scoring=physionetScorer, cv=skf)

    worker_log.info("Score:                             {}".format(np.mean(logo_scores)).ljust(92))
    worker_log.info("--------------------------------------------------------------------------------------------")

    return np.mean(logo_scores)


'''
Evaluate final model using a range of metrics

Model is evaluated using the following methods:
    - Train model on training set (used for all previous optimizations) and
    score using hidden test set
    - Run leave-one-out group cross validation, training on all but one sub
    database (a-f) for each fold
    - Run standard stratified 10-fold cross validation
'''
def scoreOptimizedModel(features, classifications, groups, train_features, test_features, train_classifications, test_classifications, optimization_fpath, feature_selection=50, **kwargs):
    # load model information from file
    with pd.HDFStore(optimization_fpath) as hdf:
        iterations = [extract_number(x)[0] for x in hdf.keys()]
        try:
            latestIteration = max(iterations)
        except ValueError:
            raise ValueError("Models have not been optimized, please run with the --optimize flag")
        latestSolution = hdf["/solution{}".format(latestIteration)]
        latestFeatures = pd.Index(hdf["/bestFeatures"])
    latestSolution = latestSolution.dropna()

    if feature_selection >= 0:
        # Use features selected by sequential feature selection algorithm
        train_features = train_features.ix[:, latestFeatures]
        test_features = test_features.ix[:, latestFeatures]
        features = features.ix[:, latestFeatures]

    # Generate scorers for each metric
    physionetScorer = MultiScorer({
        'score' : (score, {'custom_y': classifications}),
        'sensitivity' : (sensitivity, {'custom_y': classifications}),
        'specificity' : (specificity, {'custom_y': classifications})
    })
    physionetScorer2 = MultiScorer({
        'score' : (score, {'custom_y': classifications}),
        'sensitivity' : (sensitivity, {'custom_y': classifications}),
        'specificity' : (specificity, {'custom_y': classifications})
    })
    physionetScorer3 = MultiScorer({
        'score' : (score, {'custom_y': classifications}),
        'sensitivity' : (sensitivity, {'custom_y': classifications}),
        'specificity' : (specificity, {'custom_y': classifications})
    })
    # Rebuild model from saved parameters
    # TODO: Replace rebuilding with pickling of original model builds, so that
    # they can be directly loaded back from file
    train_classifications = filterNoiseSamples(train_classifications)
    test_classifications = filterNoiseSamples(test_classifications)
    classifications = filterNoiseSamples(classifications)

    model = buildModel(**latestSolution)
    model.fit(train_features, train_classifications)


    # Score model on hidden test set using custom Physionet metric
    physionetScorer(model, test_features, test_classifications)
    results = physionetScorer.get_results()
    finalScore = results['score'][0]
    sens = results['sensitivity'][0]
    spec = results['specificity'][0]

    logging.info("--------------------------------------------------------------------------------------------")
    logging.info("Hidden test set results:".ljust(92))
    logging.info("Score:                        {}".format(finalScore).ljust(92))
    logging.info("Sensitivity:                  {}".format(sens).ljust(92))
    logging.info("Specificity:                  {}".format(spec).ljust(92))
    logging.info("Selected features:".ljust(92))
    for line in textwrap.fill(" ".join([str(x) for x in test_features.columns]), width=89).split('\n'):
        logging.info(line.ljust(92))
    logging.info("--------------------------------------------------------------------------------------------")

    logo = LeaveOneGroupOut()
    skf = RepeatedStratifiedKFold(n_splits=10, random_state=42, n_repeats=10)

    # Evaluate model using leav-one-out and startified 10-fold cross-validation
    for i in xrange(10):
        cross_val_score(model, features, classifications, groups, physionetScorer2, logo)
    logo_results = physionetScorer2.get_results().copy()

    logo_scores = np.array(logo_results['score']).reshape((10, 6)).mean(axis=0)
    logo_scores_std = np.array(logo_results['score']).reshape((10, 6)).std(axis=0)
    logo_scores_mean = np.mean(logo_scores)
    logo_scores_stdd = np.std(logo_scores)
    logo_scores = [u"{0:.4f}±{1:.4f}".format(x, y) for x, y in zip(logo_scores, logo_scores_std)]
    logo_scores.append(u"{:.4f}±{:.4f}".format(logo_scores_mean, logo_scores_stdd))

    logo_sens = np.array(logo_results['sensitivity']).reshape((10, 6)).mean(axis=0)
    logo_sens_std = np.array(logo_results['sensitivity']).reshape((10, 6)).std(axis=0)
    logo_sens_mean = np.mean(logo_sens)
    logo_sens_stdd = np.std(logo_sens)
    logo_sens = [u"{0:.4f}±{1:.4f}".format(x, y) for x, y in zip(logo_sens, logo_sens_std)]
    logo_sens.append(u"{:.4f}±{:.4f}".format(logo_sens_mean, logo_sens_stdd))

    logo_spec = np.array(logo_results['specificity']).reshape((10, 6)).mean(axis=0)
    logo_spec_std = np.array(logo_results['specificity']).reshape((10, 6)).std(axis=0)
    logo_spec_mean = np.mean(logo_spec)
    logo_spec_stdd = np.std(logo_spec)
    logo_spec = [u"{0:.4f}±{1:.4f}".format(x, y) for x, y in zip(logo_spec, logo_spec_std)]
    logo_spec.append(u"{:.4f}±{:.4f}".format(logo_spec_mean, logo_spec_stdd))


    # Pretty print results to logger
    table_header = [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'Mean']

    row1 = list(logo_scores)
    row1.insert(0, 'Score')
    row2 = list(logo_spec)
    row2.insert(0, 'Specificity')
    row3 = list(logo_sens)
    row3.insert(0, 'Sensitivity')

    logo_table = tabulate([row1, row2, row3], headers=table_header, tablefmt='grid', floatfmt=".4f")
    logging.info("Leave-one-out cross-validation score:".format(logo_scores).ljust(92))
    for line in logo_table.split('\n'):
        logging.info(line.ljust(92))

    logging.debug("Running 10 repeats of 10-fold stratified cross-validation...".ljust(92))
    cross_val_score(model, features, classifications, groups, physionetScorer3, skf)
    skf_results = physionetScorer3.get_results().copy()

    skf_scores = np.array(skf_results['score']).reshape((10, 10)).mean(axis=0)
    logging.info("Stratified K-fold cross-validation score:    {}".format(np.mean(skf_scores)).ljust(92))
    skf_scores_std = np.array(skf_results['score']).reshape((10, 10)).std(axis=0)
    skf_scores_mean = np.mean(skf_scores)
    skf_scores_stdd = np.std(skf_scores)
    skf_scores = [u"{0:.4f}±{1:.4f}".format(x, y) for x, y in zip(skf_scores, skf_scores_std)]
    skf_scores.append(u"{:.4f}±{:.4f}".format(skf_scores_mean, skf_scores_stdd))

    skf_sens = np.array(skf_results['sensitivity']).reshape((10, 10)).mean(axis=0)
    skf_sens_std = np.array(skf_results['sensitivity']).reshape((10, 10)).std(axis=0)
    skf_sens_mean = np.mean(skf_sens)
    skf_sens_stdd = np.std(skf_sens)
    skf_sens = [u"{0:.4f}±{1:.4f}".format(x, y) for x, y in zip(skf_sens, skf_sens_std)]
    skf_sens.append(u"{:.4f}±{:.4f}".format(skf_sens_mean, skf_sens_stdd))

    skf_spec = np.array(skf_results['specificity']).reshape((10, 10)).mean(axis=0)
    skf_spec_std = np.array(skf_results['specificity']).reshape((10, 10)).std(axis=0)
    skf_spec_mean = np.mean(skf_spec)
    skf_spec_stdd = np.std(skf_spec)
    skf_spec = [u"{0:.4f}±{1:.4f}".format(x, y) for x, y in zip(skf_spec, skf_spec_std)]
    skf_spec.append(u"{:.4f}±{:.4f}".format(skf_spec_mean, skf_spec_stdd))

    table_header = list(np.arange(1, 11))
    table_header.append('Mean')
    row1 = list(skf_scores)
    row1.insert(0, 'Score')
    row2 = list(skf_spec)
    row2.insert(0, 'Specificity')
    row3 = list(skf_sens)
    row3.insert(0, 'Sensitivity')
    skf_table = tabulate([row1, row2, row3], headers=table_header, tablefmt='grid', floatfmt=".4f")
    for line in skf_table.split('\n'):
        logging.info(line.ljust(92))


'''
Create hidden dataset by splitting each sub-dataset, taking an equal number of
classes (normal, abnormal and unsure) from each database for the test set.
Without patient labels, it is currently not possible to stratify by patient
'''
def group_train_test_split(features, classifications, groups):
    gss = StratifiedShuffleSplit(test_size=0.33, n_splits=10, random_state=42)

    train_features = pd.DataFrame()
    test_features = pd.DataFrame()
    train_classifications = pd.DataFrame(columns=['class', 'quality'])
    test_classifications = pd.DataFrame(columns=['class', 'quality'])
    train_groups = np.array([])
    test_groups = np.array([])
    for i in xrange(np.max(groups)+1):
        g_feat = features.ix[groups == i]
        g_class = classifications.ix[groups == i]
        g_group = groups[groups == i]

        try:
            train_inds, test_inds = gss.split(g_feat, g_class['class']).next()
        except ValueError:
            raise ValueError("A database in the dataset has too few samples, at least 3 should be provided per sub-database")

        train_features = train_features.append(g_feat.ix[train_inds])
        test_features = test_features.append(g_feat.ix[test_inds])
        train_classifications = train_classifications.append(g_class.ix[train_inds])
        test_classifications = test_classifications.append(g_class.ix[test_inds])
        train_groups = np.append(train_groups, g_group[train_inds])
        test_groups = np.append(test_groups, g_group[test_inds])

    return (train_features, test_features, train_classifications, test_classifications, train_groups, test_groups)


'''
Run particle swarm optimization on model to determine the optimal set of hyper-parameters
'''
def optimizeClassifierModel(features, classifications, groups, optimization_fpath, num_evals=50, parallelize=False):
    cross_validator = StratifiedKFold(n_splits=10, random_state=42)

    def dummyWrapper(**kwargs):
        '''Dummy function created for debugging optimization quickly'''
        return (0.5, pd.Index(['diaCent', 'diaDur', 'diaFlat']))

    def optimizationWrapper(**kwargs):
        '''
        Wraps building and scoring of model for optimization algorithm
        '''
        process_name = multiprocessing.current_process().name
        worker_log = loggerops.create_logger(
            process_name,
            log_filename="{}.log".format(process_name),
            use_stream_handler=False
        )
        worker_log.propagate = False
        model = buildModel(worker_log=worker_log, **kwargs)
        scr = scoreModel(features, classifications, cross_validator, model, worker_log=worker_log, **kwargs)
        return scr
    # FIXME: Used for quickly debugging particle swarm optimization, remove for
    # production
    #optimizationWrapper = dummyWrapper

    # Define search space, providing model names and parameter ranges to search
    # for best solution
    search = {
        'cl1_algorithm': {
            'cl1_SVM': {
                'cl1_kernel': {
                    'cl1_linear': {'cl1_C': [0, 10]},
                    'cl1_rbf': {'cl1_gamma': [0, 1], 'cl1_C': [0, 10]},
                    'cl1_poly': {'cl1_degree': [2, 5], 'cl1_C': [0, 50], 'cl1_coef0': [0, 1]}
                }
            },
            'cl1_naive-bayes': None,
        },
        'cl2_algorithm': {
            'cl2_SVM': {
                'cl2_kernel': {
                    'cl2_linear': {'cl2_C': [0, 10]},
                    'cl2_rbf': {'cl2_gamma': [0, 1], 'cl2_C': [0, 10]},
                    'cl2_poly': {'cl2_degree': [2, 5], 'cl2_C': [0, 50], 'cl2_coef0': [0, 1]}
                }
            },
            'cl2_naive-bayes': None,
        },
        'cl3_algorithm': {
            'cl3_SVM': {
                'cl3_kernel': {
                    'cl3_linear': {'cl3_C': [0, 10]},
                    'cl3_rbf': {'cl3_gamma': [0, 1], 'cl3_C': [0, 10]},
                    'cl3_poly': {'cl3_degree': [2, 5], 'cl3_C': [0, 50], 'cl3_coef0': [0, 1]}
                }
            },
            'cl3_naive-bayes': None,
        },
        'lr_meta_algorithm': {
            'lr_C': [0.1, 20],
            'lr_penalty': [0, 1]

        }
    }


    # Optimization can be performed in parallel or synchronously based on user
    # preference
    if parallelize:
        pmap = optunity.pmap
    else:
        pmap = map

    # Optunity library methods used for transforming search space to a format
    # useable by the optimization algorithm
    tree = search_spaces.SearchTree(search)
    box = tree.to_box()

    # wrap the decoder and constraints for the internal search space representation
    f = tree.wrap_decoder(optimizationWrapper)
    f = constraints.wrap_constraints(f, (-sys.float_info.max, pd.Index(['test', 'test2'])), range_oo=box)

    # Create particle swarm object
    suggestion = optunity.suggest_solver(num_evals, "particle swarm", **box)
    solver = optunity.make_solver(**suggestion)
    # Apply optimization to model using formatted search space
    solution, details = optunity.optimize(
        solver,
        f,
        maximize=True,
        max_evals=num_evals,
        pmap=pmap,
        decoder=tree.decode,
        solutionFPath=optimization_fpath
    )

    # Print final optimized solution
    logging.info("Solution:".ljust(92))
    for item in solution.iteritems():
        logging.info("{:20.20}{:72.72}".format(item[0], str(item[1])))

def filterNoiseSamples(classification):
    lowQInd = classification[classification['quality'] == 0].index
    classification.ix[lowQInd, 'class'] = 0
    classification = classification['class']

    return classification


'''
Get number from string. Taken from: https://stackoverflow.com/questions/43074685/find-file-in-directory-with-the-highest-number-in-the-filename
'''
def extract_number(f):
    s = re.findall("\d+$",f)
    return (int(s[0]) if s else -1,f)

'''
Code taken from: https://stackoverflow.com/questions/250357/truncate-a-string-without-ending-in-the-middle-of-a-word

Returns a string of at most `max_length` characters, cutting only at
word-boundaries. If the string was truncated, `suffix` will be appended.
'''
def smart_truncate1(text, max_length=100, suffix='...'):

    if len(text) > max_length:
        pattern = r'^(.{0,%d}\S)\s.*' % (max_length-len(suffix)-1)
        return re.sub(pattern, r'\1' + suffix, text)
    else:
        return text
