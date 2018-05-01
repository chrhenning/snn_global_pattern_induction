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
@title           :dataset.py
@author          :ch
@contact         :christian@ini.ethz.ch
@created         :03/20/2017
@version         :1.0
@python_version  :3.5.2

This class describes a general dataset template, such that all dataset readers
have to fit the defined structure.

A dataset reader (such as the MNIST reader) has to instantiate an object of the
defined class and provide all the data to it, such that the simulation has a
unified interface to data.
"""

import configuration as config

from sklearn.model_selection import train_test_split

import logging
logger = logging.getLogger(config.logging_name)

class Dataset(object):
    """A general dataset template that is used as an interface from the
    simulator to access an arbitrary dataset (for which a reader has to be
    defined).

    In order to allow the simulator to process a different dataset, you have to
    implement an inherited class. You must always call the constructor of this
    base class first when instantiating the implemented subclass. When reading
    the dataset, you have to fille the attribute 'samples' and append
    references to the attributes 'train' and 'test' (and 'val' if needed). Each
    sample must be an instance of the subclass Sample, that is defined in this
    module. After the data is completely read, you have to specify the input
    and output size of the network. You may either do this manually or by
    calling the method '_determine_in_out_size'.

    Attributes:
        train: The training set. A list of references to Sample instances.
        test: The test set. A list of references to Sample instances.
        val: The validation set. A list of references to Sample instances.
        samples: A list of instances of the Sample class. Each instance
            represents uniquely a sample in the dataset. The list has to
            contain all samples that are included in the attribute data.
        input_size: The number of input neurons to the network.
        output_size: The number of output neurons. This equals to the number of
            labels for the particular multiclass problem.
    """

    class Sample(object):
        """This class describes a single sample in the dataset.

        Attributes:
            inputs: The input to the neural network for this sample. Must be a
                list of numbers that can be directly interpreted as firing
                rate (plain number, no Brian unit).
            label: The correct output class of the sample (ground truth).
            raw: (optional) The raw input data. This can be used for evaluation
                or visualization purposes (e.g. plotting an MNIST sample). It is
                not required by the simulator.
        """
        def __init__(self, inputs, label, raw=None):
            """Setup sample object by assigning given data to attributes.

            Args:
                inputs: A 1D float32 numpy array whose values range between
                    [config.input_fr_min, config.input_fr_max].
                label: The ground truth of the sample.
                raw: (optional) Data that can be stored for dataset specific or
                    user-specified functionalities.
            """
            self._inputs = inputs
            self._label = label
            self.raw = raw

        @property
        def inputs(self):
            """Getter for read-only attribute inputs"""
            return self._inputs

        @property
        def label(self):
            """Getter for read-only attribute label"""
            return self._label


    def __init__(self):
        # Internally, everything is stored in a certain structure, such that it
        # can easily be backuped (for instance via pickle).
        data = {}
        data.setdefault('train', [])
        data.setdefault('test', [])
        data.setdefault('val', None)

        data.setdefault('samples', [])
        data.setdefault('input_size', [])
        data.setdefault('output_size', [])

        self._data = data

    @property
    def train(self):
        """Getter for read-only attribute train"""
        return self._data['train']

    @property
    def test(self):
        """Getter for read-only attribute test"""
        return self._data['test']

    @property
    def val(self):
        """Getter for read-only attribute val"""
        return self._data['val']

    @property
    def samples(self):
        """Getter for read-only attribute samples"""
        return self._data['samples']

    @property
    def input_size(self):
        """Getter for read-only attribute input_size"""
        return self._data['input_size']

    @property
    def output_size(self):
        """Getter for read-only attribute output_size"""
        return self._data['output_size']

    def _determine_in_out_size(self):
        """Determines the input and output size of the current dataset.

        In order to determine the input size, a single sample is considered
        and its input vector size is used. The output size is computed by
        counting the number of distinct labels across all samples.

        Args:

        Returns:
        """
        assert(len(self.samples) > 0)
        assert(len(self.samples[0].inputs.shape) == 1)
        self._data['input_size'] = self.samples[0].inputs.shape[0]

        # Determine output size.
        labels = set()

        for s in self.samples:
            labels.add(s.label)

        self._data['output_size'] = len(labels)

    @staticmethod
    def _split_data(inputs, labels, split_factor):
        """Split a dataset into training and test set.

        Args:
            inputs: A python list. Some of its entries are assigned to the
                train_inputs and the remainder to the test_inputs.
            labels: Ground truth labels corresponding to the entries of inputs.
                The entries of this list must be single-value. They are used to
                sample the data in a balanced (stratified) manner.
            split_factor: The relative percentage of samples that shall be part
                of the training set. The rest will be assigned to the test set
                using stratified sampling.

        Returns:
            train_inputs: Part of the inputs list that has been assigned to the
                training set.
            train_labels: The corresponding labels entries to the train_inputs.
            test_inputs: The samples of the inputs list that have not been
                assigned to the list train_inputs.
            test_labels: The corresponding labels entries to the test_inputs.
        """
        # TODO Delete this function in case no more functionality is added and
        # replace all function calls by the corresponding sklearn method.
        assert(split_factor > 0 and split_factor < 1)

        train_inputs, test_inputs, train_labels, test_labels = \
            train_test_split(inputs, labels, train_size=split_factor, \
                             stratify=labels)

        return train_inputs, train_labels, test_inputs, test_labels

if __name__ == '__main__':
    pass


