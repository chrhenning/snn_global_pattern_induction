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
@title           :readout/classifier.py
@author          :ch
@contact         :christian@ini.ethz.ch
@created         :07/11/2017
@version         :1.0
@python_version  :3.5.2

This is an abstract base class for readout classifiers, that can assign output
firing rates to class labels.

Readout classifiers are used, if there is not one output neuron corresponding
to one class in the network. In such a case, an additional classifier is
trained on the firing rates measured during the presentation of a sample.

Classifiers should provide an API similar to scikit-learn classifiers,
implementing at least the fit and predict method.
"""

import configuration as config

from abc import ABC, abstractmethod

import logging
logger = logging.getLogger(config.logging_name)

class Classifier(ABC):
    """Readout classifier base class.

    Attributes:
    """
    def __init__(self):
        """Class constructor.

        Args:

        Returns:
        """
        pass

    @abstractmethod
    def fit(self, X, y):
        """Train the model.

        Args:
            X: A numpy array of size num_samples x num_features. I.e., the
                training input.
            y: A numpy array of size num_samples. I.e., the ground truth
                labels.

        Returns:
        """
        pass

    @abstractmethod
    def predict(self, X):
        """Predict the labels for given samples using the trained model.

        Args:
            X: A numpy array of size num_samples x num_features. I.e., the
                training input.

        Returns:
            A numpy array of size num_samples, containing the predicted labels.
                labels.
        """
        pass

if __name__ == '__main__':
    pass


