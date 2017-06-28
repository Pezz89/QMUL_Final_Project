import datetime
import pdb
import numpy as np
import scipy as sp
import scipy.stats
import warnings
import sys
from copy import deepcopy
from itertools import combinations
from collections import deque
from sklearn.metrics import get_scorer
from sklearn.base import clone
from sklearn.base import BaseEstimator
from sklearn.base import MetaEstimatorMixin
from mlxtend.externals.name_estimators import _name_estimators
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import cross_val_score

class SequentialFeatureSelectorGroup(SequentialFeatureSelector):
    def fit(self, X, y, groups=None):
        """Perform feature selection and learn model from training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object

        """

        if not isinstance(self.k_features, int) and\
                not isinstance(self.k_features, tuple):
            raise AttributeError('k_features must be a positive integer'
                                 ' or tuple')

        if isinstance(self.k_features, int) and (self.k_features < 1 or
                                                 self.k_features > X.shape[1]):
            raise AttributeError('k_features must be a positive integer'
                                 ' between 1 and X.shape[1], got %s'
                                 % (self.k_features, ))

        if isinstance(self.k_features, tuple):
            if len(self.k_features) != 2:
                raise AttributeError('k_features tuple must consist of 2'
                                     ' elements a min and a max value.')

            if self.k_features[0] not in range(1, X.shape[1] + 1):
                raise AttributeError('k_features tuple min value must be in'
                                     ' range(1, X.shape[1]+1).')

            if self.k_features[1] not in range(1, X.shape[1] + 1):
                raise AttributeError('k_features tuple max value must be in'
                                     ' range(1, X.shape[1]+1).')

            if self.k_features[0] > self.k_features[1]:
                raise AttributeError('The min k_features value must be larger'
                                     ' than the max k_features value.')

        if self.skip_if_stuck:
            sdq = deque(maxlen=4)
        else:
            sdq = deque(maxlen=0)

        if isinstance(self.k_features, tuple):
            select_in_range = True
        else:
            select_in_range = False
            k_to_select = self.k_features

        self.subsets_ = {}
        orig_set = set(range(X.shape[1]))
        if self.forward:
            if select_in_range:
                k_to_select = self.k_features[1]
            k_idx = ()
            k = 0
        else:
            if select_in_range:
                k_to_select = self.k_features[0]
            k_idx = tuple(range(X.shape[1]))
            k = len(k_idx)
            k_score = self._calc_score(X, y, k_idx, groups=groups)
            self.subsets_[k] = {
                'feature_idx': k_idx,
                'cv_scores': k_score,
                'avg_score': k_score.mean()
            }

        best_subset = None
        k_score = 0
        try:
            while k != k_to_select:
                prev_subset = set(k_idx)
                if self.forward:
                    k_idx, k_score, cv_scores = self._inclusion(
                        orig_set=orig_set,
                        subset=prev_subset,
                        X=X,
                        y=y,
                        groups=groups
                    )
                else:
                    k_idx, k_score, cv_scores = self._exclusion(
                        feature_set=prev_subset,
                        X=X,
                        y=y,
                        groups=groups
                    )

                if self.floating and not self._is_stuck(sdq):
                    (new_feature,) = set(k_idx) ^ prev_subset
                    if self.forward:
                        k_idx_c, k_score_c, cv_scores_c = self._exclusion(
                            feature_set=k_idx,
                            fixed_feature=new_feature,
                            X=X,
                            y=y,
                            groups=groups
                        )
                    else:
                        k_idx_c, k_score_c, cv_scores_c = self._inclusion(
                            orig_set=orig_set - {new_feature},
                            subset=set(k_idx),
                            X=X,
                            y=y,
                            groups=groups
                        )

                    if k_score_c and k_score_c > k_score:
                        k_idx, k_score, cv_scores = \
                            k_idx_c, k_score_c, cv_scores_c

                k = len(k_idx)
                # floating can lead to multiple same-sized subsets
                if k not in self.subsets_ or (self.subsets_[k]['avg_score'] <
                                              k_score):
                    self.subsets_[k] = {
                        'feature_idx': k_idx,
                        'cv_scores': cv_scores,
                        'avg_score': k_score
                    }

                # k_idx must be a set otherwise different permutations
                # would not be found as equal in _is_stuck
                sdq.append(set(k_idx))

                if self.verbose == 1:
                    sys.stderr.write('\rFeatures: %d/%s' % (
                        len(k_idx),
                        k_to_select
                    ))
                    sys.stderr.flush()
                elif self.verbose > 1:
                    sys.stderr.write('\n[%s] Features: %d/%s -- score: %s' % (
                        datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        len(k_idx),
                        k_to_select,
                        k_score
                    ))

                if self._TESTING_INTERRUPT_MODE:
                    raise KeyboardInterrupt

        except KeyboardInterrupt as e:
            self.interrupted_ = True
            sys.stderr.write('\nSTOPPING EARLY DUE TO KEYBOARD INTERRUPT...')

        if select_in_range:
            max_score = float('-inf')
            for k in self.subsets_:
                if self.subsets_[k]['avg_score'] > max_score:
                    max_score = self.subsets_[k]['avg_score']
                    best_subset = k
            k_score = max_score
            k_idx = self.subsets_[best_subset]['feature_idx']

        self.k_feature_idx_ = k_idx
        self.k_score_ = k_score
        self.subsets_plus_ = dict()
        self.fitted = True
        return self


    def _calc_score(self, X, y, indices, groups=None):
        if self.cv:
            scores = cross_val_score(self.est_,
                                     X[:, indices], y,
                                     cv=self.cv,
                                     scoring=self.scorer,
                                     n_jobs=self.n_jobs,
                                     pre_dispatch=self.pre_dispatch,
                                     groups=groups
                                     )
        else:
            self.est_.fit(X[:, indices], y)
            scores = np.array([self.scorer(self.est_, X[:, indices], y)])
        return scores

    def _inclusion(self, orig_set, subset, X, y, groups=None):
        all_avg_scores = []
        all_cv_scores = []
        all_subsets = []
        res = (None, None, None)
        remaining = orig_set - subset
        if remaining:
            for feature in remaining:
                new_subset = tuple(subset | {feature})
                cv_scores = self._calc_score(X, y, new_subset, groups=groups)
                all_avg_scores.append(cv_scores.mean())
                all_cv_scores.append(cv_scores)
                all_subsets.append(new_subset)
            best = np.argmax(all_avg_scores)
            res = (all_subsets[best],
                   all_avg_scores[best],
                   all_cv_scores[best])
        return res

    def _exclusion(self, feature_set, X, y, fixed_feature=None, groups=None):
        n = len(feature_set)
        res = (None, None, None)
        if n > 1:
            all_avg_scores = []
            all_cv_scores = []
            all_subsets = []
            for p in combinations(feature_set, r=n - 1):
                if fixed_feature and fixed_feature not in set(p):
                    continue
                cv_scores = self._calc_score(X, y, p, groups=groups)
                all_avg_scores.append(cv_scores.mean())
                all_cv_scores.append(cv_scores)
                all_subsets.append(p)
            best = np.argmax(all_avg_scores)
            res = (all_subsets[best],
                   all_avg_scores[best],
                   all_cv_scores[best])
        return res
