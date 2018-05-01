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
@title           :brian_wrapper.py
@author          :ch
@contact         :christian@ini.ethz.ch
@created         :04/13/2017
@version         :1.0
@python_version  :3.5.2

A wrapper for some Brian2 functions.

This wrapper should help to avoid code dublicates. E.g., when generating a
Synapses instance, we usually have to execute repeatingly the same weight
initialization.
"""

import configuration as config
from util.config_exception import ConfigException
import weight_initialization as winit

import brian2 as b2
import numpy as np
import re

import logging
logger = logging.getLogger(config.logging_name)

def neuron_group(N, model, method, threshold, reset, refractory, namespace,
                 initialization, name, layer=None):
    """Create an instance of class NeuronGroup.

    Args:
        N: See definition of class NeuronGroup.
        model: See definition of class NeuronGroup.
        method: See definition of class NeuronGroup.
        threshold: See definition of class NeuronGroup.
        reset: See definition of class NeuronGroup.
        refractory: See definition of class NeuronGroup.
        namespace: See definition of class NeuronGroup.
        initialization: A dictionary that maps from parameter names in the
            NeuronGroup to values, that this parameter should be initialized
            with.
        name: See definition of class NeuronGroup.
        layer: The index of the current layer. If defined, a variable 'layer'
            with this index will be added to the namespace.

    Returns:
        An instance of class NeuronGroup with initialized parameters as
        specified.
    """
    model, namespace = _add_layer(model, namespace, layer)

    neuron_group = b2.NeuronGroup(N, model, method=method, threshold=threshold,
                                  reset=reset, refractory=refractory,
                                  namespace=namespace, name=name)

    if initialization is None:
        return neuron_group

    for par, val in initialization.items():
        _initialize_parameter(neuron_group, par, val, neuron_group.N)

    return neuron_group

def synapses(source, target, model, method, on_pre, on_post, delay, namespace,
             initialization, name, connections=None, layer=None):
    """Create an instance of class Synapses.

    Note, all changes made later on (e.g., new connections after this method
    was called), will not be affected by the initialization.

    Args:
        target: See definition of class Synapses.
        source: See definition of class Synapses.
        model: See definition of class Synapses.
        method: See definition of class Synapses.
        on_pre: See definition of class Synapses.
        on_post: See definition of class Synapses.
        delay: See definition of class Synapses.
        initialization: A dictionary that maps from parameter names in the
            Synapses instance to values, that this parameter should be
            initialized with. (The values in the dict (if not single value)
            must match N_pre times N_post. Non-existing synapses will be
            ignored during parameter assignment.)
        namespace: See definition of class Synapses.
        name: See definition of class Synapses.
        connections: A tuple of two arrays corresponding to the parameters 'i'
            and 'j' in the method 'connect' of class Synapses. If None,
            full-connections are established.
        layer: The index of the current layer. If synapses are part of
            projections between consecutive layers, the value should be the
            index of the postsynaptic layer. If defined, a variable 'layer'
            with this index will be added to the namespace.

    Returns:
        An instance of class Synapses with initialized parameters as specified.
        If weights are not initialized due to the initialization dict, then a
        proper initialization method is chosen according to the config.
    """
    # We don't have to specify the layer for a Synapse, as it just uses the
    # variable from its postsynaptic NeuronGroup.
    #model, namespace = _add_layer(model, namespace, layer)

    synapses = b2.Synapses(source, target, model=model, method=method,
                           on_pre=on_pre, on_post=on_post, delay=delay,
                           namespace=namespace, name=name)

    # Connections must be established before weights can be initialized.
    # TODO instead of establishing connections here, give the user an wrapper
    # for the connect method, that applies initialization. (i.e., an own
    # Synapses class)
    if connections is None:
        synapses.connect()
    else:
        synapses.connect(i=connections[0], j=connections[1])

    init_weights = True

    if initialization is not None:
        for par, val in initialization.items():
            _initialize_parameter(synapses, par, val, (synapses.N_pre,
                                                       synapses.N_post))

        if 'w' in initialization.keys():
            init_weights = False

    if init_weights:
        if config.weight_initialization == 'uniform':
            val = lambda size: winit.uniform(size,
                                             wmin=config.winit_uniform_wmin,
                                             wmax=config.winit_uniform_wmax)
            _initialize_parameter(synapses, 'w', val, (synapses.N_pre,
                                                       synapses.N_post))
        else:
            raise ConfigException('The weight initialization method \'%s\' ' \
                                  % config.weight_initialization + 'is ' \
                                  + 'unknown')

    return synapses

def _add_layer(model, namespace, layer):
    """Add the variable layer to the model.

    Args:
        model: The model that shall define a Brian object.
        namespace: The namespace, passed to the initialization of a Brian
            object.
        layer: The layer, in which the object will be located. If this
            parameter is None, then nothing is changed.

    Returns:
        model: Extended with the variable 'layer', if needed.
        namespace: The same namespace, except that layer is now set correctly.
    """
    if layer is not None:
        eq = 'layer : 1 (constant, shared)'
        if model is None:
            model = eq
        elif isinstance(model, str):
            # Assert, that variable has not yet been defined.
            layer_defined = False
            for line in model.split('\n'):
                if re.match('^layer\W', line):
                    layer_defined = True
                    break
            if not layer_defined:
                model += '\n' + eq
        else:
            assert(isinstance(model, b2.Equations))
            if not 'layer' in model.names:
                model += eq

        if namespace is None:
            namespace = {}
        namespace['layer'] = layer

    return model, namespace

def _initialize_parameter(instance, parameter, value, size):
    """Initialize (or set) a parameter of an Brian2 instance with a certain
    value.

    Args:
        instance: An instance of a Brian2 class, that has modifiable
            parameters (E.g., NeuronGroup or Synapses).
        parameter: The name of the parameter to modify as string.
        value: The value that should be assigned to the parameter. If not
            single value (i.e., no shared parameter), it must be a tensor that
            hold the full dimensionality of the instance. If value is not
            callable and not a single value, it must be a numpy array.
        size: A tuple of integers defining the size of parameters. This
            parameter is ignored if value if not callable (i.e., no function).

    Returns:
    """
    if callable(value):
        val = value(size)
    else:
        val = value

    if isinstance(val, np.ndarray):

        if isinstance(instance, b2.Synapses):
            # In a fully connected layer we would have synaptic weights as a
            # matrix N_pre x N_post, which are stored in a Synapses object as a
            # list of length N_pre*N_post. The index i,j (Connection from
            # presynaptic neuron i to postsynaptic neuron j) can then be
            # accessed via index i*N_post+j.
            getattr(instance, parameter)[instance.i*instance.N_post + \
                instance.j] = val[instance.i, instance.j]
        else:
            getattr(instance, parameter)[:] = val

        # Here is a more general, but very slow solution.
        """
        # Note, np.ndenumerate will delete the units (but will scale them, such
        # that 20mV are resolving to v=0.02).
        unit = getattr(instance, parameter).unit

        for index, v in np.ndenumerate(val):
            if len(index) == 1:
                i = np.squeeze(index)
            else:
                i = index
            a = getattr(instance, parameter)[i]
            # If attribute exists (e.g., synapse).
            if a.size > 0:
                getattr(instance, parameter)[i] = v * unit
        """
    else:
        setattr(instance, parameter, val)

if __name__ == '__main__':
    pass


