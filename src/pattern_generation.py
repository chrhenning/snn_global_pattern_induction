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
@title           :pattern_generation.py
@author          :ch
@contact         :christian@ini.ethz.ch
@created         :03/30/2017
@version         :1.0
@python_version  :3.5.2

The generation of patterns that should form the activation of excitatory
neurons in hidden layers and the output layer.

The class implemented in this module will only generate the pattern according
to the user configuration. It will not install the pattern influence to the
network.
It is important, that the class already knows the complete network structure
(pattern is generated after the network has been setup), since later pattern
implementation might have to consider interlayer dependencies and therefore
have to take the connection structure in between layers into account.
"""

import configuration as config
from util.config_exception import ConfigException
from util import lin_alg
from util import simplex_construction as simplex
from pypatterns.observable import Observable

import numpy as np
from scipy.sparse import random as sparse_random
from scipy.stats import norm as normal
from sklearn.preprocessing import normalize

import logging
logger = logging.getLogger(config.logging_name)

class PatternGeneration(Observable):
    """Generate preferred activation patterns for excitatory neurons in hidden
    layers as well as in the output layer.

    Note, that the pattern of the output layer always will be clamped to
    canonical unit vectors, i.e. the output layer activations for different
    classes are always perfectly orthogonal. All other layers get an activation
    pattern assigned for each output class according to the settings in the
    config file.

    This class notifies all observers everytime, the internal patterns are
    modified.

    Attributes:
    """
    def __init__(self, network):
        """Initiate the pattern generation according to the user configs.

        Args:
            network: An instance of the class NetworkModel, that fully defines
                the number of excitatory neurons per layer and there
                connectivity.

        Returns:
        """
        super().__init__()

        self._network = network

        self._num_patterns = network.data.output_size

        # Will contain a numpy matrix of size layer_size_i x num_patterns for
        # each layer i starting at i equals 1 (omit input layer).
        self._patterns = []

        # For how many layers do we need to generate a pattern? For all hidden
        # layers plus the output layer. But if we do not want to use a
        # classifier to readout the output, then we simply use an orthogonal
        # pattern for the output. Hence, in this case we only want to generate
        # a pattern for all hidden layers according to the config and use an
        # fully orthogonal pattern for the output.
        self._patterns_to_generate = network.num_layers-2 # num hidden layers
        if config.output_size is not None:
            self._patterns_to_generate += 1

        if config.pattern_type == 'random':
            self._random_pattern()
        elif config.pattern_type == 'sparse_random':
            self._sparse_random_pattern()
        elif config.pattern_type == 'dense_orthogonal':
            self._dense_orthogonal_pattern()
        elif config.pattern_type == 'sparse_blurred':
            self._sparse_blurred_pattern()
        elif config.pattern_type == 'sparse_threshold':
            self._sparse_threshold_pattern()
        elif config.pattern_type == 'max_mutual_equidistant':
            self._max_mutual_equidistant_pattern()
        else:
            raise ConfigException('The chosen pattern type \'%s\' is ' \
                                  % (config.pattern_type) + 'unknown. Please' \
                                  + ' reconsider the option \'pattern_type\'.')

        # Add pattern of output layer pattern (identity matrix), if no extra
        # readout classifier shall be used.
        if config.output_size is None:
            self._patterns.append(np.identity(self._num_patterns,
                                              dtype=np.float32))

        # This flag shall make sure, that we do not unintentionally scale the
        # pattern such that it adopts its layer-wise influence more than once.
        self._layer_wise_decay_applied = False

    @property
    def num_patterns(self):
        """Number of distinct patterns.

        The number of patterns equals to the number of output neurons.

        Args:

        Returns:
            Number of patterns.
        """
        return self._num_patterns

    def get_pattern(self, layer, output=None):
        """Get the pattern of a layer belonging to a specific output class.

        Args:
            layer: The index of the layer the pattern should be taken from,
                starting at 1.
            output: The index of the output class. If not passed, the matrix
                with all patterns will be returned (one pattern per column).

        Returns:
            The desired output activation pattern for the given layer and
            output class.
            Returns None for invalid indices.
        """
        if layer >= 1 and layer < self._network.num_layers:
            if output is not None:
                if output >= 0 and output < self.num_patterns:
                    return self._patterns[layer-1][:,output]
                else:
                    return None
            else:
                return self._patterns[layer-1]
        else:
            None

    def invert_patterns(self):
        """This method inverts the patterns, that have been defined for an
        output class.

        This method works as follows. For a given pattern x, it determines
        x_m = max(x). Then it uses the normalized version of
            x_m * ones(x.shape) - x
        as inverted pattern.

        NOTE, this method has to be performed before any scaling or similar is
        applied to the pattern, as it normalizes the modified patterns.

        Args:

        Returns:
        """
        for i, pattern in enumerate(self._patterns):
            for j in range(self.num_patterns):
                x = pattern[:,j]
                x_max = x.max()
                x_new = x_max * np.ones(x.shape) - x
                self._patterns[i][:,j] = x_new / np.linalg.norm(x_new)

        # Notify observers.
        self.update_observers(self.__class__.__name__)


    def apply_layer_wise_decay(self):
        """The influence of the pattern on lower layers should not be as strong
        as its influence on higher layers. Otherwise, the network is unlikely
        to learn the input distribution. Therefore, this method adopts the
        decay as specified by config.pattern_layer_influence.

        Args:

        Returns:
        """
        if self._layer_wise_decay_applied:
            logger.warning("Layer-wise decay on patterns has been applied ' \
                           + 'more than once.")

        self._layer_wise_decay_applied = True

        influence = config.pattern_layer_influence
        if isinstance(influence, str):
            if influence == 'linear':
                influence = lambda l, L: l/(L-1)
            elif influence == 'quadratic':
                influence = lambda l, L: (l/(L-1))**2
            else:
                raise ConfigException('Method ' + influence + 'unknown for ' \
                                      + 'config option \'' \
                                      + 'pattern_layer_influence\'.')

        for i, pattern in enumerate(self._patterns):
            scaler = influence(i+1, self._network.num_layers)
            self._patterns[i] = pattern * scaler

        # Notify observers.
        self.update_observers(self.__class__.__name__)

    def scale_strength(self):
        """Scale pattern strength according to the option
        config.pattern_strength.

        This method simply multiplies the patterns according to the scalar
        defined in the config. If an array of scalars is defined, then every
        pattern is scaled by the scaling factor thats associated with its
        output class.

        Args:

        Returns:
        """
        factors = config.pattern_strength
        if not isinstance(factors, list):
            factors = [factors] * (self.num_patterns)
        # TODO ConfigExceptions for these kind of assertions.
        assert(len(factors) == self.num_patterns)

        for i, pattern in enumerate(self._patterns):
            for j in range(self.num_patterns):
                self._patterns[i][:,j] = pattern[:,j] * factors[j]

        # Notify observers.
        self.update_observers(self.__class__.__name__)

    def decay_influence(self):
        """At the end of training, the classifier should be able to clearly
        distinguish output classes without the need of an external drive, that
        stimulates certain patterns. In this respect, the influence of the
        external pattern should decay during the training, as the neuronal
        assemblies already should be strong enough.

        This method decays the pattern according to the config option
        pattern_induction_decay.

        Args:

        Returns:
        """
        scalar = config.pattern_induction_decay
        logger.debug("Patterns are decayed by %g." % \
                     config.pattern_induction_decay)

        for i, pattern in enumerate(self._patterns):
            self._patterns[i] = pattern * scalar

        # Notify observers.
        self.update_observers(self.__class__.__name__)

    def _random_pattern(self):
        """Generate a random pattern for each hidden layer and output class.

        Args:

        Returns:
        """
        for i in range(1, 1+self._patterns_to_generate):
            ap = np.random.rand(self._network.layer_size(i),
                                self.num_patterns)
            ap = normalize(ap, norm='l2', axis=0, copy=False)
            self._patterns.append(ap)

    def _sparse_random_pattern(self):
        """Generate a sparse random pattern for each hidden layer and output
        class.

        Args:

        Returns:
        """
        densities = config.pattern_sparse_random_density
        if not isinstance(densities, list):
            densities = [densities] * (self._patterns_to_generate)

        for i in range(1, 1+self._patterns_to_generate):
            ap = sparse_random(self._network.layer_size(i), self.num_patterns,
                               density=densities[i-1])
            ap = np.array(ap.todense())
            ap = normalize(ap, norm='l2', axis=0, copy=False)
            self._patterns.append(ap)

    def _dense_orthogonal_pattern(self):
        """Generate a random dense pattern for each hidden layer and output
        class. Within a layer, the activity patterns of different classes shall
        be orthogonal, i.e. the random matrix will be orthogonalized using
        Gram-Schmidt.

        Args:

        Returns:
        """
        for i in range(1, 1+self._patterns_to_generate):
            ap = np.random.rand(self._network.layer_size(i),
                                self._num_patterns)
            ap = lin_alg.gram_schmidt(ap)
            n, m = ap.shape
            if m > n:
                ap = lin_alg.extend_orthogonal_base(ap)

            ap = normalize(ap, norm='l2', axis=0, copy=False)
            self._patterns.append(ap)

    def _sparse_blurred_pattern(self):
        """Construct an orthogonal matrix NxM with canonical unit vectors as
        columns (if M>N, then extend the basis with the method
        lin_alg.extend_orthogonal_base at the end). The unit vectors are then
        blurred according to a bell curve in the neighborhood of their 1 entry
        (except for the extended ones).

        Args:

        Returns:
        """
        std_devs = config.pattern_sparse_blurred_std_dev
        if not isinstance(std_devs, list):
            std_devs = [std_devs] * (self._patterns_to_generate)
        for i in range(1, 1+self._patterns_to_generate):
            ap = np.zeros((self._network.layer_size(i),
                           self.num_patterns))
            n, m = ap.shape

            # Equally sample 1 entries.
            ones = []
            if m >= n:
                ones = list(range(n))
            else:
                # k is number of zeros in between ones. Split k into m-1 bins.
                k = n-m
                dist = int(k/(m-1))
                rem = k % (m-1)
                ones.append(0)
                for o in range(1,m):
                    ones.append(ones[o-1] + dist + (1 if rem > 0 else 0) + 1)
                    rem -= 1
            for c in range(min(n,m)):
                # Vector containing blurry distances to the current 1 entry.
                # Thus, each entry is drawn from the interval [d-0.5, d+0.5],
                # where d is the distance to the 1 entry.
                d = [j-ones[c] for j in range(n)] + (np.random.rand(n)-0.5)
                ap[:,c] = normal.pdf(d, scale=std_devs[i-1])
            if m > n:
                ap = lin_alg.extend_orthogonal_base(ap)

            ap = normalize(ap, norm='l2', axis=0, copy=False)
            self._patterns.append(ap)

    def _sparse_threshold_pattern(self):
        """Generate a pattern according to the method
        _dense_orthogonal_pattern. Afterwards, we set small values to zero
        (even those under a certain threshold or a certain percentage of
        entries).

        Args:

        Returns:
        """
        thresholds = config.pattern_sparse_threshold_thld
        if not isinstance(thresholds, list):
            thresholds = [thresholds] * (self._patterns_to_generate)
        use_percentage = isinstance(thresholds[0], str)
        if use_percentage:
            thresholds = [float(t) for t in thresholds]

        self._dense_orthogonal_pattern()

        for i in range(1, 1+self._patterns_to_generate):
            ap = self._patterns[i-1]
            ap = lin_alg.introduce_sparcity(ap, thresholds[i-1],
                                            use_percentage)
            ap = normalize(ap, norm='l2', axis=0, copy=False)
            self._patterns[i-1] = ap

    def _max_mutual_equidistant_pattern(self):
        """Generate patterns, such that the mutual euclidean distance between
        patterns is maximized.

        The method utilizes the construction of a simplex with mutual
        equidistant points.

        Args:

        Returns:
        """
        for i in range(1, 1+self._patterns_to_generate):
            n = self._network.layer_size(i)
            m = self._num_patterns

            if m > n+1:
                raise ConfigException('The pattern ' \
                                      + '\'_max_mutual_equidistant\' does ' \
                                      + 'not allow hidden layers with more ' \
                                      + 'than one additional neuron compared' \
                                      + 'to the output layer.')

            ap = simplex.get_unit_simplex(n,m)
            self._patterns.append(ap)

if __name__ == '__main__':
    pass


