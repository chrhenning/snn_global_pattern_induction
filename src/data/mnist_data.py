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
@title           :mnist_data.py
@author          :ch
@contact         :christian@ini.ethz.ch
@created         :03/15/2017
@version         :1.0
@python_version  :3.5.2

Read the MNIST dataset into an appropriate internal representation.
"""

import configuration as config
from .dataset import Dataset

from os import path

import struct
import numpy as np
import time
import _pickle as pickle

import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(config.logging_name)

class MNISTData(Dataset):
    """An instance of the class shall represent the MNIST dataset.

    Attributes: (additional to baseclass)
    """
    def __init__(self):
        """Read the MNIST digit classification dataset from file.

        This method checks whether the dataset has been read before (a pickle
        dump has been generated). If so, it reads the dump. Otherwise, it
        reads the data from scratch and creates a dump for future usage.

        Args:

        Returns:
        """
        super().__init__()

        start = time.time()

        logger.info('Reading MNIST dataset ...')

        # If data has been processed before.
        if path.isfile(config.mnist_pickle_dump):
            with open(config.mnist_pickle_dump, 'rb') as f:
                self._data = pickle.load(f)

        else:
            # read labels
            train_labels = MNISTData._read_labels(config.mnist_train_label)
            test_labels = MNISTData._read_labels(config.mnist_test_label)

            # read images
            train_inputs = MNISTData._read_images(config.mnist_train_images)
            test_inputs = MNISTData._read_images(config.mnist_test_images)

            # Bring these raw readings into the internal structure of the
            # Dataset class
            assert(len(train_labels) == len(train_inputs))
            assert(len(test_labels) == len(test_inputs))

            # Generate a list of training samples
            for i, raw_img in enumerate(train_inputs):
                label = train_labels[i]
                sample = MNISTData._generate_sample(raw_img, label)
                self.samples.append(sample)
                self.train.append(sample)

            # Generate a list of test samples
            for i, raw_img in enumerate(test_inputs):
                label = test_labels[i]
                sample = MNISTData._generate_sample(raw_img, label)
                self.samples.append(sample)
                self.test.append(sample)

            # Compute input and output number of neurons.
            self._determine_in_out_size()

            # Save read dataset to allow faster reading in future.
            with open(config.mnist_pickle_dump, 'wb') as f:
                pickle.dump(self._data, f)

        end = time.time()
        logger.info('Elapsed time to read dataset: %f sec' % (end-start))

    @staticmethod
    def _read_labels(filename):
        """Reading a set of labels from a file.

        Args:
            filename: Path and name of the byte file that contains the labels.

        Returns:
            A list of labels.
        """
        assert(path.isfile(filename))

        logger.info('Reading labels from %s.' % filename)
        with open(filename, "rb") as f:
            # Skip magic number.
            _ = f.read(4)
            # Get number of labels in this file.
            num = int.from_bytes(f.read(4), byteorder='big')
            logger.info('Number of labels in current file: %d' % num)

            labels = []

            i = 0
            byte = f.read(1)
            while byte:
                i += 1

                label = struct.unpack('B', byte)[0]
                labels.append(label)

                byte = f.read(1)

            assert(i == num)

            return labels

    @staticmethod
    def _read_images(filename):
        """Reading a set of images from a file.

        Args:
            filename: Path and name of the byte file that contains the images.

        Returns:
            A list of images. Each image will be a 2D numpy array of type uint8.
        """
        assert(path.isfile(filename))

        logger.info('Reading images from %s.' % filename)
        with open(filename, 'rb') as f:
            # Skip magic number
            _ = f.read(4)
            # Get number of images in this file.
            num = int.from_bytes(f.read(4), byteorder='big')
            logger.info('Number of images in current file: %d' % num)
            # Get number of rows and columns.
            rows = int.from_bytes(f.read(4), byteorder='big')
            cols = int.from_bytes(f.read(4), byteorder='big')

            images = []
            curr_img = None

            i = 0
            byte = f.read(1)
            while byte:
                # If the current byte marks the beginning of a new image.
                if i % (rows*cols) == 0:
                    curr_img = np.zeros((rows,cols), dtype=np.uint8)
                    images.append(curr_img)

                intensity = struct.unpack('B', byte)[0]
                # Compute row and column offset of current pixel.
                s = i % (rows*cols)
                r = s // cols
                c = s % cols
                curr_img[r][c] = intensity

                i += 1
                byte = f.read(1)

            assert(len(images) == num)

            return images

    @staticmethod
    def _generate_sample(raw_image, label):
        """Generates a Sample instance from the given raw data.

        Args:
            raw_image: A 2D numpy array of type uint8 containing the raw
                gray-scale pixel information.
            label: The ground truth of the sample.
        Returns:
            A valid instance of the class Sample. The raw pixel data will be
            kept in the designated Sample attribute.
        """
        inputs = raw_image.flatten().astype(np.float32)
        # Scale inputs [0,255] to desired firing rate range.
        # z in [x,y] -> c in [a,b]:
        # c = (z-x)/(y-x) * (b-a) + a
        inputs = inputs/255. * (config.input_fr_max - config.input_fr_min) \
                 + config.input_fr_min

        return Dataset.Sample(inputs, label, raw=raw_image)

    @staticmethod
    def plot_sample(sample, interactive=False, file_name=None):
        """Plot a single MNIST sample.

        This method is thought to be helpful for evaluation and debugging
        purposes.

        Args:
            sample: An instance of the class Sample, that has been generated by
                the MNISTData class (it must contain the raw image data).
            interactive: Turn on interactive mode. Thus program will run in
                background while figure is displayed. The figure will be
                displayed until another one is displayed, the user closes it or
                the program has terminated. If this option is deactivated, the
                program will freeze until the user closes the figure.
            file_name: (optional) If a file name is provided, then the image
                will be written into a file instead of plotted to the screen.

        Returns:
        """
        plt.title('Label of shown sample: %d' % sample.label)
        plt.axis('off')
        if interactive:
            plt.ion()
        plt.imshow(sample.raw)
        if file_name is not None:
            plt.savefig(file_name, bbox_inches='tight')
        else:
            plt.show()

if __name__ == '__main__':
    pass

