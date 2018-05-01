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
@title           :hidden_layer.py
@author          :ch
@contact         :christian@ini.ethz.ch
@created         :03/15/2017
@version         :1.0
@python_version  :3.5.2

This module contains a helper class to generate the excitatory and
inhibitory neurons of a hidden layer as well as their connections.

The name of the module is a bit misleading, as the module specifies a gneral
processing layer, that (if not otherwise specified) does account to the output
layer as well. Such a processing layer should be viewed in contrast to an input
layer (i.e. Poisson layer), whose behavior is extrinsical defined.
However, the NetworkModel could also define another output model instead of
using this one (i.e. other neural and/or synaptic dynamics).

This module is implemented as a thread, since layers can be setup in parallel.
"""

import configuration as config
import brian_wrapper as bw

import brian2 as b2
import numpy as np
import threading

import logging
logger = logging.getLogger(config.logging_name)

class HiddenLayer(threading.Thread):
    """Thread to setup a processing layer.

    Attributes:
        exc_neurons: A NeuronGroup representing the excitatory neurons of the
            layer.
        inh_neurons: A NeuronGroup representing the inhibitory neurons of the
            layer.
        ei_synapses: A Synapses instance representing the connections from
            excitatory to inhibitory neurons.
        ie_synapses: A Synapses instance representing the connections from
            inhibitory to excitatory neurons.
    """
    def __init__(self, num_neurons, layer_index, window_size, layer_name):
        """Initialize thread.

        Args:
            num_neurons: Number of excitatory resp. inhibitory neurons in this
                layer.
            layer_index: The index of the layer (e.g. input layer has index 0).
            window_size: This is either a single number or None. If a single
                number is specified, then it denotes to the number of
                neighboring neurons to each side of an exc. neuron to which an
                inh. neuron should connect. The config option
                'lateral_inhibition_window' provides a detailed description.
            layer_name: A unique name for the layer. Will be used to form
                NeuronGroup and Synapses layer.

        Returns:
        """
        threading.Thread.__init__(self)

        self._num_neurons = num_neurons
        self._layer_index = layer_index
        self._window_size = window_size
        self._layer_name = layer_name

    def run(self):
        """Create a processing layer.

        A processing layer consists of excitatory and inhibitory neurons. Each
        excitatory neuron is connected to a corresponding inhibitory neuron.
        Depending on the parameter 'window_size', this inhibitory neuron will
        introduce lateral inhibition to the neughborhood of its incoming
        excitatory neuron.

        Args:

        Returns:
        """
        eq = config._equation_module
        # To access the correct set of equations, we have to substract 1 from
        # the current layer index, as the input layer has no equations.
        l = self._layer_index - 1

        self._exc_neurons = bw.neuron_group(self._num_neurons, eq.ne_model[l],
                                            eq.ne_method[l],
                                            eq.ne_threshold[l], eq.ne_reset[l],
                                            eq.ne_refractory[l],
                                            eq.ne_namespace[l],
                                            eq.ne_initialization[l],
                                            name='neurons_exc_'+ \
                                                self._layer_name,
                                            layer=self._layer_index)
        self._inh_neurons = bw.neuron_group(self._num_neurons, eq.ni_model[l],
                                            eq.ni_method[l],
                                            eq.ni_threshold[l],eq.ni_reset[l],
                                            eq.ni_refractory[l],
                                            eq.ni_namespace[l],
                                            eq.ni_initialization[l],
                                            name='neurons_inh_'+ \
                                                self._layer_name,
                                            layer=self._layer_index)

        # Create 'ei' connections (one-to-one connections).
        ei_connections = (np.arange(self._num_neurons),
                          np.arange(self._num_neurons))

        # Create 'ie' connections according to window parameter.
        ie_connections = ([], [])
        if config.lateral_inhibition_window is None:
            # Connect each inh. neuron to all exc. neurons except the one with
            # the same index.
            for n in range(self._num_neurons):
                # All neuron indices except of the current one.
                outgoing = np.concatenate((np.arange(0,n),
                                           np.arange(n,self._num_neurons)),
                                          axis=0)
                ie_connections[0].extend([n]*len(outgoing))
                ie_connections[1].extend(outgoing)
        else:
            # Connect inh. neuron n to the 'window_size' neighbors on the left
            # and right of exc. neuron n.
            for n in range(self._num_neurons):
                left = np.arange(max(0, n-self._window_size), n)
                right = np.arange(n+1, min(self._num_neurons,
                                           n+self._window_size+1))
                outgoing = np.concatenate((left,right), axis=0)

                ie_connections[0].extend([n]*len(outgoing))
                ie_connections[1].extend(outgoing)

        self._ei_synapses = bw.synapses(self._exc_neurons, self._inh_neurons,
                                        eq.ei_model[l], eq.ei_method[l],
                                        eq.ei_on_pre[l], eq.ei_on_post[l],
                                        eq.ei_delay[l], eq.ei_namespace[l],
                                        eq.ei_initialization[l],
                                        name='synapses_ei_'+self._layer_name,
                                        connections=ei_connections,
                                        layer=self._layer_index)
        self._ie_synapses = bw.synapses(self._inh_neurons, self._exc_neurons,
                                        eq.ie_model[l], eq.ie_method[l],
                                        eq.ie_on_pre[l], eq.ie_on_post[l],
                                        eq.ie_delay[l], eq.ei_namespace[l],
                                        eq.ie_initialization[l],
                                        name='synapses_ie_'+self._layer_name,
                                        connections=ie_connections,
                                        layer=self._layer_index)

    @property
    def exc_neurons(self):
        """The NeuronGroup that represents the excitatory neurons of this
        layer.
        """
        return self._exc_neurons

    @property
    def inh_neurons(self):
        """The NeuronGroup that represents the inhibitory neurons of this
        layer.
        """
        return self._inh_neurons

    @property
    def ei_synapses(self):
        """A Synapse object containing the connections from exc_neurons to
        inh_neurons.
        """
        return self._ei_synapses

    @property
    def ie_synapses(self):
        """A Synapse object containing the connections from inh_neurons to
        exc_neurons.
        """
        return self._ie_synapses

if __name__ == '__main__':
    pass


