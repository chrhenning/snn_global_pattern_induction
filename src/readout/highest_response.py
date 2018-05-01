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
@title           :readout/highest_response.py
@author          :ch
@contact         :christian@ini.ethz.ch
@created         :07/11/2017
@version         :1.0
@python_version  :3.5.2

Assigning classes to output firing patterns based on the activity of neurons
that have been shown to be high responsive to certains input stimuli during
training.

Each output neuron will be assigned to be responsive to a certain class. This
class will be the one, where it showed the highest average firing rate during
presentations of stimuli belonging to that class. During classification, the
average firing rate of all neurons assigned to a class is taken. The highest of
these average firing rates defines the predicted class.
"""

import configuration as config
from readout.classifier import Classifier

import numpy as np

import logging
logger = logging.getLogger(config.logging_name)

class HighestResponse(Classifier):
    """A readout classifier that uses the highest response hypothesis to
    predict class labels from output firing rates.

    The mechanism is described in:
        'Unsupervised learning of digit recognition using spike-timing-dependent
        plasticity', Diehl Peter, Cook Matthew, 2015 (10.3389/fncom.2015.00099)

    Attributes:
    """
    def __init__(self):
        """Init empty classifier.

        Args:

        Returns:
        """
        self._assignments = None
        self._num_classes = None
        self._assigned_classes = None

    def fit(self, X, y, num_classes=None):
        """Train the model.

        Args:
            X: A numpy array of size num_samples x num_features. I.e., the
                training input.
            y: A numpy array of size num_samples. I.e., the ground truth
                labels.
            num_classes: The number of classes. If not specified, then the
                maximum label in y (plus one)  is taken.

        Returns:
        """
        if num_classes is None:
            num_classes = np.unique(y).max() + 1
        self._num_classes = num_classes

        num_outputs = X.shape[1]
        # Class with maximum avg firing rate is stored for each neuron.
        max_fr = np.zeros(num_outputs)
        # Class index per neuron that causes the maximum firing rate above.
        assignments = np.zeros(num_outputs)

        for i in range(num_classes):
            outputs_i = X[np.where(y == i)[0]]
            if outputs_i.size == 0:
                logger.warning('No sample for class %d in recent training ' \
                               % i + 'batch')
                continue

            # Average firing rates of output neurons for samples from class i.
            avg_fr_i = np.sum(outputs_i, axis=0)/outputs_i.shape[0]

            highest_response = np.where(avg_fr_i > max_fr)[0]
            max_fr[highest_response] = avg_fr_i[highest_response]
            assignments[highest_response] = i

        self._assigned_classes = np.unique(assignments)
        if self._assigned_classes.size != num_classes:
            logger.warning('Not all classes have been assigned to ' \
                           + 'output neurons.')

        self._assignments = assignments

    def predict(self, X):
        """Predict the labels for given samples using the trained model.

        Args:
            X: A numpy array of size num_samples x num_features. I.e., the
                training input.

        Returns:
            A numpy array of size num_samples, containing the predicted labels.
                labels.
        """
        assert(self._assignments is not None)
        assignments = self._assignments

        # Average fr rate of assigned neurons per class and per sample.
        avg_fr_per_class = np.zeros((X.shape[0], self._num_classes))

        for i in self._assigned_classes:
            ii = int(i)
            # Output firing rates assigned to current class.
            class_outputs = X[:,np.where(assignments == i)[0]]
            avg_fr_per_class[:,ii] = np.sum(class_outputs, axis=1) \
                    / class_outputs.shape[1]

        return np.argmax(avg_fr_per_class, axis=1)

if __name__ == '__main__':
    pass


