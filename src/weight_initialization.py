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
@title           :weight_initialization.py
@author          :ch
@contact         :christian@ini.ethz.ch
@created         :04/12/2017
@version         :1.0
@python_version  :3.5.2

A collection of methods that can be used to initialize weights.

Each function defined in this module should have one positional argument (a
tuple of dimensions). All other arguments must be optional (assigned with
default values). All the functions return a tensor (defined by the input
tuple).

Note, all methods must return a matrix of type numpy.ndarray.
"""

import numpy as np

def uniform(size, wmin=0, wmax=1):
    """Return a random matrix with uniformly distributed entries between wmin
    and wmax.

    Args:
        size: Tuple of ints, containing dimensions.
        wmin: Lower bound for entries in the returned matrix.
        wmax: Upper bound for entries in the returned matrix.

    Returns:
        A uniform random matrix.
    """
    return np.random.random(size) * (wmax-wmin) + wmin

if __name__ == '__main__':
    pass


