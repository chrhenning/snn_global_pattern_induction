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
@title           :util/utils.py
@author          :ch
@contact         :christian@ini.ethz.ch
@created         :03/30/2017
@version         :1.0
@python_version  :3.5.2

This module contains an assembly of helper functions with relevant and distinct
functionality. Please refer to the individual function descriptions for
details.
"""

import random
import logging
import os
import sys

def config_logger(name, log_file, file_level, console_level):
    """Configure the logger that should be used by all modules in this
    package.

    This method sets up a logger, such that all messages are written to console
    and to an extra logging file. Both outputs will be the same, except that
    a message logged to file contains the module name, where the message comes
    from.

    Args:
        name: The name of the created logger.
        log_file: Path of the log file.
        file_level: Log level for logging to log file.
        console_level: Log level for logging to console.

    Returns:
        The configured logger.
    """
    file_formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s' \
                                       + ' - %(module)s - %(message)s', \
                                       datefmt='%m/%d/%Y %I:%M:%S %p')
    stream_formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s' \
                                         + ' - %(message)s', \
                                         datefmt='%m/%d/%Y %I:%M:%S %p')

    # Check if directory of logfile already exists.
    if not os.path.isdir(os.path.dirname(log_file)):
        os.mkdir(os.path.dirname(log_file))
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(file_level)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(stream_formatter)
    stream_handler.setLevel(console_level)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

def yield_chunks(arr, n):
    """This method yields chunks of size n from the array arr.

    The original purpose to this method is, to split up lists into chunks of
    size 'num_threads'. However, it can be used for any other pupose as well.

    Args:
        arr: The list to split into chunks.
        n: Size of each chunk (up to that size, last one might be smaller).

    Returns:
        Returns a generator instance.

    """
    for i in range(0, len(arr), n):
        yield arr[i:i + n]

def list_to_val_dependent_chunks(lst, partition):
    """Split a list into value-dependent chunks. I.e., a chunk lst[n] ...
    lst[m] should span an interval smaller equal a given partition size.

    For example: A list of timesteps might be split into chunks spanning the
    same duration, e.g. 2 seconds.

    Args:
        lst: The list to split.
        partition: The interval each chunk may span.

    Returns:
        A generator producing the desired chunks.
    """
    limit = partition
    chunk = []
    i = 0
    while i < len(lst):
        chunk.append(lst[i])
        if lst[i] >= limit:
            yield chunk
            chunk = []
            limit += partition
        i += 1
    if len(chunk) > 0:
        yield chunk

def list_to_val_dependent_slices(lst, partition):
    """The same as the method list_to_val_dependent_chunks, but instead of
    actually computing the chunks, it computes the start and end index of each
    slice.

    Args:
        lst: The list to split.
        partition: The interval each chunk may span.

    Returns:
        A generator producing the desired slices (start_ind, end_ind).
    """
    limit = partition
    i = 0
    start_ind = i
    while i < len(lst):
        if lst[i] >= limit:
            yield (start_ind, i)
            start_ind = i+1
            limit += partition
        i += 1
    i -= 1
    if i > start_ind:
        yield (start_ind, i)

def random_shuffle_loop(arr):
    """Returns a generator that yields elements from the randomly shuffled
    given array. After a whole sweep through the array, the array is shuffled
    again.

    Args:
        The array to yield elements from.

    Returns:
        A generator object, that yields elements from arr.
    """
    len_arr = len(arr)
    indices = list(range(len_arr))
    i = len_arr
    while True:
        if i == len_arr:
            i = 0
            random.shuffle(indices)
        yield arr[indices[i]]
        i += 1

def set_tuple_item(tup, index, item):
    """Tuples are immutable. Therefore, this method provides item assigned to a
    copy of the initial tuple.

    Args:
        tup: The tuple, that should be mutated.
        index: The index of the item in tup that should be mutated.
        item: The value that should be assigned to tup[index].

    Returns:
        A copy of tup, but the value of one item has changed in the desired
        way.
    """
    lst = list(tup)
    lst[index] = item
    return tuple(lst)

def list_to_str(lst):
    """Convert a list of items into an underscore separated string.

    For Example: [1,2,3] -> _1_2_3

    Args:
        lst: The list to convert into a string.

    Returns:
        The stringified list.
    """
    s = ''
    for item in lst:
        s += '_%s' % (str(item))
    return s

if __name__ == '__main__':
    pass


