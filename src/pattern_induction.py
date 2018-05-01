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
@title           :pattern_induction.py
@author          :ch
@contact         :christian@ini.ethz.ch
@created         :04/06/2017
@version         :1.0
@python_version  :3.5.2

Induce a pattern into the network.

This class uses an Instance of the class NetworkModel and an instance of the
class PatternGeneration to manipulate the network such that the patterns are
stimulated during simulation.

One easy way to induce the pattern in the network, is to generate a
periodically firing neuron for each output class. This neuron is connected to
all excitatory neurons with a synaptic weight proportional to the pattern value
(no synapses for zero values). However, that synaptic weight will be
additionally scaled, depending on the distance to the output layer.
"""

import configuration as config
from util.config_exception import ConfigException
from pypatterns.observer import Observer
from eq_state_vars import EqStateVars

import brian2 as b2
import numpy as np

import logging
logger = logging.getLogger(config.logging_name)

class PatternInduction(Observer):
    """This class modifies an instance of class NetworkModel, such that a
    pattern, defined by an instance of class PatternGeneration, can be induced
    to the Brian network.

    The class observes the instance of class PatternGeneration.

    Attributes:
        active_output: The index of the output, that is currently stimulated
            (or None if no pattern stimulation takes place).
    """
    def __init__(self, network, patterns):
        """Induce pattern into network as specified in config.

        Args:
            network: An instance of the class NetworkModel.
            patterns: An instance of the class PatternGeneration.

        Returns:
        """
        super().__init__()

        self._network = network
        self._patterns = patterns
        self._eq_state = EqStateVars()

        self._active_output = None

        if config.pattern_induction_method == 'simple':
            self._simple_induction()
        elif config.pattern_induction_method == 'impact_modulation':
            self._impact_modulation_induction()
        else:
            raise ConfigException('The chosen induction_method \'%s\' is ' \
                                  % (config.pattern_induction_method) + \
                                  'unknown. Please reconsider the option ' \
                                  + '\'pattern_induction_method\'.')

        self._patterns.register(self)
        self._eq_state.register(self)

    @property
    def active_output(self):
        """The currently active output.

        Returns:
            The index of the output class that is currently considered as
            active, i.e., whose pattern is induced to the network.
            If None, no pattern is currently induced.
        """
        return self._active_output

    def _change_pattern(self, output):
        """Define which pattern (according to a certain output class), that
        shall currently be active, i.e., induced to the network.

        This method is essential to control the pattern induction to the
        network. Before it is called the first time, no pattern does
        influence the network at all.

        Args:
            output: The index of the output class that is associated with the
                current sample. If None, then there will be no pattern
                influence anymore. For instance, during sample presentations to
                the network or during the testing phase, one should pass None
                to this method.

        Returns:
        """
        # No change necessary.
        if self.active_output == output:
            return

        self._active_output = output

        if config.pattern_induction_method == 'simple':
            self._simple_change()
        elif config.pattern_induction_method == 'impact_modulation':
            self._impact_modulation_change()
        else:
            # At this point, the config given by the user has been already
            # checked.
            raise Exception('Unexpected exception. Please report')

    def update(self, *args, **kwargs):
        """Update pattern induction according to changes in the pattern itself
        or a changing output class.

        Args:

        Returns:
        """
        if args[0] == 'PatternGeneration':
            #logger.debug('Pattern induction is updated due to changes in ' \
            #             + 'observed object.')

            if config.pattern_induction_method == 'simple':
                self._simple_update()
            elif config.pattern_induction_method == 'impact_modulation':
                self._impact_modulation_update()
            else:
                # At this point, the config given by the user has been already
                # checked.
                raise Exception('Unexpected exception. Please report')
        elif args[0] == 'EqStateVars':
            if kwargs['attribute'] == 'presenting':
                # If the network is not in training mode (not plastic), then we
                # set active output to None all the time.
                if not self._eq_state.plastic:
                    self._change_pattern(None)
                # If no sample is presented, then output should also be None.
                elif not kwargs['new_value']:
                    self._change_pattern(None)
                else:
                    self._change_pattern(self._eq_state.output)
        else:
            assert(False)

    def _simple_induction(self):
        """Inducing the pattern into the network by inserting a neuron for each
        output class that connects to all excitatory neurons with non-zeros
        pattern activation.

        This pattern generates a neuron for each output class, that connects to
        all excitatory neurons in all layers except for the input layer. The
        connection weights are defined by the pattern * times the specified
        influence weight. There are no synapses for zero-connections. The
        neurons fire periodically (Poisson-neurons with fixed firing rate).

        Note, this method only sets up the induction. No pattern influence will
        yet appear. The method _change_pattern has to evoke the pattern
        induction first.

        Args:

        Returns:
        """
        num_outputs = self._patterns.num_patterns

        # Poisson neurons to stimulate pattern formation.
        pattern_neurons = b2.NeuronGroup(num_outputs, 'rates : Hz',
                                         threshold='rand()<rates*dt',
                                         name='neurons_pattern_simple')

        pattern_neurons.rates = 0 * b2.Hz
        self._network.add_component(pattern_neurons)

        self._simple_pattern_neurons = pattern_neurons

        # The pattern influence equation. I.e., at the moment, the weight
        # defined in the pattern is just added to the membrane voltage of the
        # targeted exc. neuron.
        influence_eq = 'v_post += w'

        # Set Connections.
        self._simple_pattern_synapses = []

        # Note, this loop is not parallelized due to known problems with
        # parallelizing Brian code in class NetworkModel.
        for l in range(1,self._network.num_layers):
            exc_neurons = self._network.exc_neurons(l)

            synapses = b2.Synapses(pattern_neurons, exc_neurons,
                                   model='w:volt', on_pre=influence_eq,
                                   name='synapses_pattern_simple_%d' % l)

            pattern = self._patterns.get_pattern(l)

            for o in range(num_outputs):
                for e in range(exc_neurons.N):
                    p = pattern[e,o]
                    if p != 0:
                        synapses.connect(i=o, j=e)

            self._simple_pattern_synapses.append(synapses)

        # Set conncetion weighs.
        self._simple_update()

        self._network.add_component(self._simple_pattern_synapses)

    def _impact_modulation_induction(self):
        """In this induction method, the pattern scales the influence of
        synaptic modification on the postsynaptic potential according to the
        pattern strength. The exact way of modification is defined by the user
        due to the variable 'patmod'.

        This method simply ensures, that the variable 'patmod' is part of at
        least one synaptic state.

        Note, that the default value for patmod is set in the constructor of
        EqStateVars.

        Args:

        Returns:
        """
        patmod_considered = False
        for l in range(1, self._network.num_layers):
            _, _, ei, ie, ee = self._network.brian_objects(l)
            for synapses in [ei,ie,ee]:
                if 'patmod' in synapses.get_states().keys():
                    patmod_considered = True
                    break
            if patmod_considered:
                break
        if not patmod_considered:
            error_msg = 'Induction method %s requires consideration of ' \
                    % (config.pattern_induction_method) + 'synaptic variable'\
                    + ' \'patmod\'.'
            raise ConfigException(error_msg)

    def _simple_change(self):
        """Adapt to a changing output class according to method
        _change_pattern.

        This method simply adjusts the firing rate of the neurons inducing the
        pattern according to the induction method 'simple'. I.e., setting all
        firing rates to zero except the active one.

        Args:

        Returns:
        """
        firing_rate = config.pattern_simple_fr

        rates = np.zeros(self._patterns.num_patterns)

        if self.active_output is not None:
            rates[self.active_output] = firing_rate

        self._simple_pattern_neurons.rates = rates * b2.Hz

    def _impact_modulation_change(self):
        """Adapt to a changing output class according to method
        _change_pattern.

        This method sets 'patmod' to its default value if no active output is
        present. Otherwise, it will set 'patmod' according to the pattern of
        the output class.

        Args:

        Returns:
        """
        if self.active_output is None:
            self._eq_state.patmod = config.impact_modulation_default
        else:
            patterns = []
            for l in range(1, self._network.num_layers):
                patterns.append(self._patterns.get_pattern(l, \
                    output=self.active_output))
            self._eq_state.patmod = patterns

    def _simple_update(self):
        """Updating the simple induction synapses due to changes in the
        pattern.

        This method simply sets the weights of all synapses according to the
        current pattern.

        Args:

        Returns:
        """
        for l in range(1,self._network.num_layers):
            synapses = self._simple_pattern_synapses[l-1]

            pattern = self._patterns.get_pattern(l)

            for o in range(synapses.N_pre):
                for e in range(synapses.N_post):
                    p = pattern[e,o]
                    if synapses.w[o,e].size != 0:
                        synapses.w[o,e] = p * b2.volt
                    else:
                        # If the initial value was unequal zero, a sznapse
                        # should have been created. (A pattern value cannot get
                        # non-zero.)
                        assert(p==0)

    def _impact_modulation_update(self):
        """Since for each presentation, the pattern is taken from the actual
        class PatternGeneration, no action has to be taken when these patterns
        change (we always work with the most up-to-date patterns).

        Args:

        Returns:
        """
        pass

if __name__ == '__main__':
    pass

