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
@title           :seven_segment_data.py
@author          :ch
@contact         :christian@ini.ethz.ch
@created         :03/28/2017
@version         :1.0
@python_version  :3.5.2

Read the 7segment dataset into an appropriate internal representation.

The 7segment dataset provides noisy binary states of the LEDs from a 7-segment
display. These binary input vectors (size: 7) are mapped onto digits (10 output
classes).
The precise file format is as follows. Each line represents a sample, which has
the layout "0,1,2,3,4,5,6,C", where the numbers 0 to 6 represent a binary
variable displaying the state of an LED (0 - off, 1 - on). The identifier "C"
represents the displayed digit (10 classes for digits 0 - 9).
The LED encoding is as follows:

 ---0---
|       |
1       2
|       |
 ---3---
|       |
4       5
|       |
 ---6---

"""

import configuration as config
from .dataset import Dataset

from os import path

import numpy as np
import time

import matplotlib.pyplot as plt
from matplotlib import collections as col

import logging
logger = logging.getLogger(config.logging_name)

class SevenSegmentData(Dataset):
    """An instance of the class shall represent the 7segment dataset.

    Attributes: (additional to baseclass)
    """
    def __init__(self):
        """Read the 7segment dataset from file.

        Args:

        Returns:
        """
        super().__init__()

        start = time.time()

        logger.info('Reading 7segment dataset ...')

        # Read samples from file.
        inputs, labels = SevenSegmentData._read_data(config.seven_segment_data)

        train_inputs, train_labels, test_inputs, test_labels = \
            Dataset._split_data(inputs, labels, config.seven_segment_split)

        # Bring these raw readings into the internal structure of the
        # Dataset class
        assert(len(train_labels) == len(train_inputs))
        assert(len(test_labels) == len(test_inputs))

        # Generate a list of training samples
        for i, raw_states in enumerate(train_inputs):
            label = train_labels[i]
            sample = SevenSegmentData._generate_sample(raw_states, label)
            self.samples.append(sample)
            self.train.append(sample)

        # Generate a list of test samples
        for i, raw_states in enumerate(test_inputs):
            label = test_labels[i]
            sample = SevenSegmentData._generate_sample(raw_states, label)
            self.samples.append(sample)
            self.test.append(sample)

        # Compute input and output number of neurons.
        self._determine_in_out_size()

        end = time.time()
        logger.info('Elapsed time to read dataset: %f sec' % (end-start))

    @staticmethod
    def _read_data(filename):
        """Reading the dataset from file into a list of binary vectors and a
        list of corresponding labels.

        Args:
            filename: Path and name of the file that contains the labels.

        Returns:
            inputs: A list of binary vectors, representing the LED state during
                the displaying of a digit.
            labels: List of corresponding ground-truth labels to inputs.
        """
        assert(path.isfile(filename))

        logger.info('Reading samples from %s.' % filename)
        with open(filename, "r") as f:

            inputs = []
            labels = []

            for line in f:
                sample = line.strip().split(',')
                assert(len(sample) == 8)
                sample = list(map( int, sample))

                inputs.append(sample[:-1])
                labels.append(sample[-1])

            logger.info('Number of samples in file: %d' % (len(inputs)))

            return inputs, labels

    @staticmethod
    def _generate_sample(states, label):
        """Generates a Sample instance from the given raw data.

        Args:
            states: A binary list of size 7 representing LED states.
            label: The ground truth of the sample.
        Returns:
            A valid instance of the class Sample. The raw state data will be
            kept in the designated Sample attribute.
        """
        inputs = np.array(states, dtype=np.float32)
        # Scale inputs [0,1] to desired firing rate range.
        inputs = inputs * (config.input_fr_max - config.input_fr_min) \
                 + config.input_fr_min

        return Dataset.Sample(inputs, label, raw=states)

    @staticmethod
    def plot_sample(sample, interactive=False, file_name=None):
        """Plot a single 7segment sample.

        This method is thought to be helpful for evaluation and debugging
        purposes.

        Args:
            sample: An instance of the class Sample, that has been generated by
                the SevenSegmentData class (it must contain the raw states
                data).
            interactive: Turn on interactive mode. Thus program will run in
                background while figure is displayed. The figure will be
                displayed until another one is displayed, the user closes it or
                the program has terminated. If this option is deactivated, the
                program will freeze until the user closes the figure.
            file_name: (optional) If a file name is provided, then the image
                will be written into a file instead of plotted to the screen.

        Returns:
        """
        # Define the coordinates of a line for each LED.
        led_coords = {
            0: [(.25,1.),(.75,1.)],
            1: [(.25,.5),(.25,1.)],
            2: [(.75,.5),(.75,1.)],
            3: [(.25,.5),(.75,.5)],
            4: [(.25,0.),(.25,.5)],
            5: [(.75,0.),(.75,.5)],
            6: [(.25,0.),(.75,0.)]
        }

        # Add LED lines that are active in current sample.
        leds = []
        for i, s in enumerate(sample.raw):
            if s:
               leds.append(led_coords[i])

        if interactive:
            plt.ion()
        lc = col.LineCollection(leds, linewidths=2)

        fig, ax = plt.subplots()
        ax.add_collection(lc)
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_axis_off()

        plt.title('Label of shown sample: %d' % sample.label)

        if file_name is not None:
            plt.savefig(file_name, bbox_inches='tight')
        else:
            plt.show()

if __name__ == '__main__':
    pass

