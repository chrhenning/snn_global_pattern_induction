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
@title           :network_model.py
@author          :ch
@contact         :christian@ini.ethz.ch
@created         :03/15/2017
@version         :1.0
@python_version  :3.5.2

The class in this module assembles the neural network using Brian2.

The network is composed of several layers as defined in the config file. The
first layer is a Poisson layer. All other layers (including the output layer)
are defined by supplementary modules, defining a certain processing layer. So
far, only one processing layer is defined (class HiddenLayer).
"""

import configuration as config
from util.config_exception import ConfigException
from hidden_layer import HiddenLayer
import brian_wrapper as bw
from util import utils
import equation_preparation
from pypatterns.observer import Observer
from eq_state_vars import EqStateVars
from recordings import Recordings

import brian2 as b2
import numpy  as np

import logging
logger = logging.getLogger(config.logging_name)


class NetworkModel(Observer):
    """The network structure is defined in this class.

    An object of this class will build the network as specified and can then be
    used for simulations.

    This class observes the instance of EqStateVars in order to update equation
    parameters.

    Attributes:
        data: An instance of the class Dataset. The network will be
            specifically generated to fit this dataset (size of input and
            number of classes is defined by the dataset).
        num_layers: The number of network layers.
        network: An instance of the Brian class Network. It contains all
            components of the network and should therefore be the only access
            point to run simulations (Do not use the run method of the
            MagicNetwork, as this network does not contain all components)!
    """
    def __init__(self, data):
        """Builds the network.

        Args:
            data: An instance of the class Dataset. The constructor actually
                only needs to know the input and output number of neurons from
                the dataset. However, we directly pass the whole Dataset
                instance, such that we don't have to pass it anymore in future
                function calls.

        Returns:
        """
        self._data = data

        # Determine number of hidden layers.
        num_hlayers = config.num_hidden_layers

        if isinstance(config.hidden_layer_sizes, list):
            num_hlayers = len(config.hidden_layer_sizes)

        # Check if configurations are consistent.
        if isinstance(config.lateral_inhibition_window, list) \
           and len(config.lateral_inhibition_window) != num_hlayers + 1:
            raise ConfigException('The length of the option list ' \
                                  + '\'lateral_inhibition_window\' does not' \
                                  + ' match the number of layers specified.')

        # Determine size of each layer.
        self._layer_sizes = []

        self._layer_sizes.append(data.input_size)
        if isinstance(config.hidden_layer_sizes, list):
            self._layer_sizes.extend(config.hidden_layer_sizes)
        else:
            self._layer_sizes.extend([config.hidden_layer_sizes] * num_hlayers)

        # Output layer size. The output layer has either as size the number of
        # classes or the user-defined size (if specified).
        if config.output_size is None:
            self._layer_sizes.append(data.output_size)
        else:
            self._layer_sizes.append(config.output_size)

        # Check and prepare equations
        equation_preparation.prepare_equations(num_hlayers+1)

        # To compute firing rates of neurons, we have to store the spike counts
        # of neurons before the input rates have been changed (see above
        # comment). The input layer can be ommitted.
        self._exc_prev_spike_count = []
        self._inh_prev_spike_count = [None]
        for i in range(len(self._layer_sizes)):
            self._exc_prev_spike_count.append(np.zeros(self._layer_sizes[i],
                                                       dtype=np.int64))
            if i > 0:
                self._inh_prev_spike_count.append( \
                    np.zeros(self._layer_sizes[i], dtype=np.int64))

        # In order to compute firing rates, we need to know the time difference
        # between the current time and a reference time.
        self._prev_simulation_time = np.float32(b2.defaultclock.t_)
        assert(self._prev_simulation_time == 0)

        # Layer-wise SpikeMonitors for exc. neurons.
        self._exc_spike_monitors = []
        # Layer-wise SpikeMonitors for inh. neurons (None for input layer).
        self._inh_spike_monitors = []
        # Excitatory NeuronGroup of each layer.
        self._exc_layer = []
        # Inhibitory NeuronGroup of each layer (will be None for input layer).
        self._inh_layer = []
        # Feed-forward connections from excitatory neurons of one layer to the
        # next one (fully-connected).
        self._ee_synapses = []
        # Excitatory to inhibitory connections within layer.
        self._ei_synapses = []
        # Inhibitory to excitatory connections within layer.
        self._ie_synapses = []

        ### Input Layer
        # The input of the network will be a Poisson Layer.
        self._input_group = b2.NeuronGroup(self._layer_sizes[0], 'rates : Hz',
                                           threshold='rand()<rates*dt',
                                           name='neurons_poisson_0')
        self._exc_layer.append(self._input_group)
        exc_sm_args, _ = Recordings.get_spike_monitor_args(0)
        self._exc_spike_monitors.append(b2.SpikeMonitor(self._input_group, \
            variables=exc_sm_args[0], record=exc_sm_args[1]))
        self._inh_layer.append(None)
        self._inh_spike_monitors.append(None)

        # There are no recurrent connections within the input layer.
        self._ei_synapses.append(None)
        self._ie_synapses.append(None)

        ### Hidden Layer + Output Layer
        # We can generate a seperate threade to setup each layer, as the setup
        # can be done independently.
        threads = []

        for i in range(num_hlayers + 1):
            if isinstance(config.lateral_inhibition_window, list):
                k = config.lateral_inhibition_window[i]
            else:
                k = config.lateral_inhibition_window

            threads.append(HiddenLayer(self._layer_sizes[i+1], i+1,  k,
                                       str(i+1)))

        if config.num_threads > 1:
            logger.warning('Multithreading during Network Initialization' + \
                           ' has been disabled due to known issues.')
        thread_chunks = utils.yield_chunks(threads, 1)
        #thread_chunks = utils.yield_chunks(threads, config.num_threads)

        for tc in thread_chunks:
            logger.debug('Starting threads to create %d layer/s in parallel.' \
                         % (len(tc)))
            for thread in tc:
                thread.start()

            for thread in tc:
                thread.join()

                exn = thread.exc_neurons
                inn = thread.inh_neurons
                eis = thread.ei_synapses
                ies = thread.ie_synapses

                l = len(self._exc_spike_monitors)
                exc_sm_args, inh_sm_args = Recordings.get_spike_monitor_args(l)

                self._exc_layer.append(exn)
                self._exc_spike_monitors.append(b2.SpikeMonitor(exn, \
                    variables=exc_sm_args[0], record=exc_sm_args[1]))
                self._inh_layer.append(inn)
                self._inh_spike_monitors.append(b2.SpikeMonitor(inn, \
                    variables=inh_sm_args[0], record=inh_sm_args[1]))

                self._ei_synapses.append(eis)
                self._ie_synapses.append(ies)

        ### Connect layers.
        for i in range(self.num_layers - 1):
            # Connect excitatory neurons of layer i with those of later i+1.
            eq = config._equation_module
            ees = bw.synapses(self._exc_layer[i], self._exc_layer[i+1],
                              eq.ee_model[i], eq.ee_method[i], eq.ee_on_pre[i],
                              eq.ee_on_post[i], eq.ee_delay[i],
                              eq.ee_namespace[i],
                              eq.ee_initialization[i],
                              name='synapses_ee_'+str(i+1),
                              connections=None, # Fully-connected
                              layer=i+1)
            self._ee_synapses.append(ees)

        ### Create the Brian simluation control center (Network)
        self._network = b2.Network()
        # Add all components to the network.
        self._network.add(self._exc_layer)
        self._network.add(self._inh_layer[1:])
        self._network.add(self._exc_spike_monitors)
        self._network.add(self._inh_spike_monitors[1:])
        self._network.add(self._ee_synapses)
        self._network.add(self._ei_synapses[1:])
        self._network.add(self._ie_synapses[1:])
        # Double-check correctness if one changes the code!
        #print(self._network.objects)

        # FIXME delete assertions
        assert(len(self._exc_layer) == self.num_layers)
        assert(len(self._inh_layer) == self.num_layers)
        assert(len(self._exc_spike_monitors) == self.num_layers)
        assert(len(self._inh_spike_monitors) == self.num_layers)
        assert(len(self._ei_synapses) == self.num_layers)
        assert(len(self._ie_synapses) == self.num_layers)

        self._eq_state = EqStateVars()
        self._eq_state.register(self)

    @property
    def num_layers(self):
        """The number of layers in the network (including input and output
        layer).

        Args:

        Returns:
            The number of network layers.
        """
        return len(self._layer_sizes)

    @property
    def network(self):
        """Returns a Brian object of type Network, that contains all components
        of the network. Only this object should be used for simulations!

        Args:

        Returns:
            A network object that should be used to run simulations.
        """
        return self._network

    @property
    def data(self):
        """Returns the Dataset object that defines the data, that is simulated
        with this network.

        Args:

        Returns:
            The attribute data.
        """
        return self._data

    def add_component(self, component):
        """Add a component to the network.

        Add a Brian2 instance to the network. This method is thought as
        an interface to artificially perturb (e.g. pattern induction) or to
        record from the network. Hence, this method should never be used to
        change the network structure, as the methods in this class (and
        subclasses) define the network graph for visualization as well as other
        computations.

        Args:
            component: A Brian 2 instance or a list of Brian2 instances.

        Returns:
        """
        self._network.add(component)

    def layer_size(self, layer):
        """Get the number of excitatory resp. inhibitory neurons of a specific
        layer.

        Args:
            layer: The index of network layer (ranges from 0 to num_layers-1,
                where index 0 represents the input layer).

        Returns:
            The number of excitatory neurons in a certain layer. Note, that a
            layer has as many inhibitory neurons as excitatory ones. So the
            total number of neurons in a layer is twice as much as the size
            returned by this method, except for the input layer.
            If layer is an invalid index, then the method returns None.
        """
        if layer >= 0 and layer < self.num_layers:
            return self._layer_sizes[layer]
        else:
            return None

    def exc_spike_monitor(self, layer):
        """Get the SpikeMonitor of excitatory neurons in a specific layer.

        Args:
            layer: The index of network layer (ranges from 0 to num_layers-1,
                where index 0 represents the input layer).

        Returns:
            An instance of a spike monitor, that monitors the spiking events of
            all excitatory neurons in a certain layer.
            If layer is an invalid index, then the method returns None.
        """
        if layer >= 0 and layer < self.num_layers:
            return self._exc_spike_monitors[layer]
        else:
            return None

    def inh_spike_monitor(self, layer):
        """Get the SpikeMonitor of inhibitory neurons in a specific layer.

        Args:
            layer: The index of network layer (ranges from 1 to num_layers-1,
                where index 1 represents the first hidden layer). Note, that
                the input layer has no inhibitory neurons.

        Returns:
            An instance of a spike monitor, that monitors the spiking events of
            all inhibitory neurons in a certain layer.
            If layer is an invalid index, then the method returns None.
        """
        if layer >= 1 and layer < self.num_layers:
            return self._inh_spike_monitors[layer]
        else:
            return None

    def exc_neurons(self, layer):
        """Get the excitatory neurons in a specific layer.

        Args:
            layer: The index of network layer (ranges from 1 to num_layers-1).

        Returns:
            The NeuronGroup instance, representing the excitatory neurons of a
            layer.
            If layer is an invalid index, then the method returns None.
        """
        if layer >= 1 and layer < self.num_layers:
            return self._exc_layer[layer]
        else:
            return None

    def brian_objects(self, layer):
        """Get the Brian instances of a layer that define the underlying
        network structure.

        Deprecated: This interface is likely to change in future.

        Args:
            layer: The index of network layer (ranges from 0 to num_layers-1,
                where index 0 represents the input layer).

        Returns:
            exn: Excitatory neurons of the layer (Poisson neurons for layer 0).
            inn: Inhibitory neurons of the layer (None for layer 0).
            eis: Exc. to inh. synapses of layer (None for layer 0).
            ies: Inh. to exc. synapses of layer (None for layer 0).
            ees: Incoming exc. to exc. connections from previous layer (None
                 for layer 0).
        """
        if layer >= 0 and layer < self.num_layers:
            ees = self._ee_synapses[layer-1] if layer > 0 else None
            return self._exc_layer[layer], self._inh_layer[layer], \
                   self._ei_synapses[layer], self._ie_synapses[layer], ees
        else:
            return None


    def set_input_rates(self, rates, reset_fr=True):
        """Set the firing rates of input neurons.

        Note, this will also reset the layer-wise firing rates, meaning if
        firing rates are requested in future, they'll be computed relative to
        the current simulation state. See method exc_firing_rates for details.

        Args:
            rates: The firing rates that should be assigned to neurons. Must be
                a 1D vector having the size of the input layer. (Should be the
                inputs attribute of class Sample)
            reset_fr: Whether or not to reset the current firing rates. If
                True, then firing rates will be computet relative to the curren
                simulation time in future.

        Returns:

        """
        self._input_group.rates = rates * b2.Hz

        if reset_fr:
            # Update previous spike counts, such that the firing rates are
            # computed relative to the current point in time.
            for i in range(len(self._layer_sizes)):
                # Total spike counts so far.
                self._exc_prev_spike_count[i] = np.array(
                    self._exc_spike_monitors[i].count)
                if i > 0:
                    self._inh_prev_spike_count[i] = np.array(
                        self._inh_spike_monitors[i].count)

            # Store the current time for firing rate computation.
            self._prev_simulation_time = np.float32(b2.defaultclock.t_)

    def _firing_rates(self, spike_monitor, reference_counts):
        """Compute firing rates. This is only a helper method.

        See 'exc_firing_rates' for details.

        Args:
            spike_monitor: A spike monitor, that can be used to compute the
                overall spike counts of a layer.
            reference_counts: The spike counts that have been accumulated until
                the time point 'self._prev_simulation_time'.

        Returns:
            The computed firing rate, relative to 'self._prev_simulation_time'.
        """
        spike_counts = np.array(spike_monitor.count)
        duration = np.float32(b2.defaultclock.t_) - \
                   self._prev_simulation_time
        return (spike_counts - reference_counts) / duration

    def output_spike_counts(self):
        """Return the spike counts of the excitatory output neurons.

        Spike counts are measured relative to the last point in time where the
        firing rates have been reset.

        Args:

        Returns:
            A vector of ints, referring to the number of output spikes
            generated per output neuron.
        """
        total_spike_counts = \
            np.array(self._exc_spike_monitors[self.num_layers-1].count)
        reference_counts = self._exc_prev_spike_count[self.num_layers-1]
        return total_spike_counts - reference_counts

    def exc_firing_rates(self, layer):
        """Get the firing rates of excitatory neurons in a layer.

        The firing rates are measured since the last time the input rates have
        been reset. Actually, the firing rates should be measured for the time
        a sample has been presented as stimuli to the network. If this method
        is called, after the simulation of a sample has finished, this kind of
        behavior is achieved. However, this method allows more flexibility.

        Args:
            layer: The index of network layer (ranges from 0 to num_layers-1)

        Returns:
            A float32 numpy array of firing rates for each neuron in the layer.
            If layer is an invalid index, then the method returns None.
        """
        if layer >= 0 and layer < self.num_layers:
            return self._firing_rates(self._exc_spike_monitors[layer], \
                                      self._exc_prev_spike_count[layer])
        else:
            return None

    def inh_firing_rates(self, layer):
        """Get the firing rates of inhibitory neurons in a layer.

        See 'exc_firing_rates' for details.

        Args:
            layer: The index of network layer (ranges from 1 to num_layers-1,
                where index 1 represents the first hidden layer). Note, that
                firing rates for the input layer are not measured.

        Returns:
            A float32 numpy array of firing rates for each neuron in the layer.
            If layer is an invalid index, then the method returns None.
        """
        if layer >= 1 and layer < self.num_layers:
            return self._firing_rates(self._inh_spike_monitors[layer], \
                                      self._inh_prev_spike_count[layer])
        else:
            return None

    def update(self, *args, **kwargs):
        """Update the parameter values of state variables defined in synaptic
        resp. neuron equations.

        Args:

        Returns:
        """
        if args[0] == 'EqStateVars':
            attr = kwargs['attribute']
            val = kwargs['new_value']

            for obj in self._network.objects:
                if isinstance(obj, b2.NeuronGroup) or isinstance(obj,
                                                                 b2.Synapses):
                    if attr in obj.get_states().keys():
                        # The attribute 'patmod' is treated differently, if the
                        # default value is not provided. See EqStateVars for
                        # details.
                        # Note, instead of resetting patmod on every iter
                        # (slow), one could just reset a single gate per
                        # synapse (such that patmod only has to be updated once
                        # the pattern changes).
                        # E.g., replace patmod in all equations with:
                        #   [(default-patmod)*g + patmod]
                        if attr == 'patmod' and isinstance(val, list):
                            assert(isinstance(obj, b2.Synapses) and \
                                   hasattr(obj, 'layer'))
                            l = getattr(obj, 'layer')
                            l = int(0 + l)
                            pattern = val[l]
                            getattr(obj,attr)[obj.i*obj.N_post+obj.j] = \
                                    pattern[obj.j]

                        else:
                            setattr(obj, attr, val)
        else:
            assert(False)

if __name__ == '__main__':
    pass


