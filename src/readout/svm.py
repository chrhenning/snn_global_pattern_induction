#!/usr/bin/env python3
# Copyright 2017 Christian Henning
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
@title           :readout/svm.py
@author          :ch
@contact         :christian@ini.ethz.ch
@created         :07/11/2017
@version         :1.0
@python_version  :3.5.2

A Support Vector Classifier, that predicts class labels for given output rates.
"""

import configuration as config
from readout.classifier import Classifier

from sklearn import svm
from sklearn.model_selection import GridSearchCV

import logging
logger = logging.getLogger(config.logging_name)

class SVM(Classifier):
    """A readout classifier implementation that uses the SVC implementation
    from Scikit-Learn to classify output firing rates.

    Attributes:
    """
    def __init__(self):
        """Initialize empty model.

        Args:

        Returns:
        """
        self._svm = None

    def fit(self, X, y, C=1.0, kernel='rbf', decision_function_shape='ovr'):
        """Train the model.

        Args:
            X: A numpy array of size num_samples x num_features. I.e., the
                training input.
            y: A numpy array of size num_samples. I.e., the ground truth
                labels.
            C: Parameter C of sklearn.svm.SVC. May be an array of values, in
                which case a grid-search would be applied.
            kernel: Parameter from sklearn.svm.SVC.
            decision_function_shape: Parameter from sklearn.svm.SVC.

        Returns:
        """
        clf = svm.SVC(decision_function_shape=decision_function_shape,
                      kernel=kernel)
        if isinstance(C,list):
            par_grid = [{'C': C}]
            gdCV = GridSearchCV(clf, par_grid, n_jobs=config.num_threads)
            self._svm = gdCV
        else:
            self._svm = clf
        self._svm.fit(X,y)

    def predict(self, X):
        """Predict the labels for given samples using the trained model.

        Args:
            X: A numpy array of size num_samples x num_features. I.e., the
                training input.

        Returns:
            A numpy array of size num_samples, containing the predicted labels.
                labels.
        """
        assert(self._svm is not None)
        return self._svm.predict(X)

if __name__ == '__main__':
    pass


