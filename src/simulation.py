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
@title           :simulation.py
@author          :ch
@contact         :christian@ini.ethz.ch
@created         :05/03/2017
@version         :1.0
@python_version  :3.5.2

This package contains an observable wrapper for the Brian run method. This
could be observed by other classes, e.g. classes updating online visualizations
every time the run method is called.
"""

import configuration as config
from pypatterns.observable import Observable
from pypatterns.singleton import Singleton
from eq_state_vars import EqStateVars
import evaluation

import brian2 as b2
from collections import defaultdict
import time
import numpy as np

import logging
logger = logging.getLogger(config.logging_name)

class Simulation(Observable, metaclass=Singleton):
    """This class essentially represents a wrapper for the Brian2 run method,
    such that simulations can be easier managed, when processing batches.

    Attributes:
        network: An instance of class NetworkModel.
        profiler: Dictionary mapping from network object names to runtimes
            (i.e., times needed to simulate those objects.)
    """
    def __init__(self):
        """Just create object.

        Args:

        Returns:
        """
        super().__init__()

        self._network = None
        self._eq_state = EqStateVars()

        if config.simulation_profiling:
            self._profiler = defaultdict(float)
        else:
            self._profiler = None

    @property
    def profiler(self):
        """Getter for attribute profiler.

        Args:

        Returns:
            A dict is profiling is enabled. None otherwise.
        """
        return self._profiler

    @property
    def network(self):
        """The instance of class NetworkModel, that is simulated.

        Args:

        Returns:
        """
        return self._network

    @network.setter
    def network(self, value):
        """Setter for the attribute network.

        Args:
            value: A NetworkModel instance.
        """
        self._network = value

    def simulate_batch(self, batch, ident, epoch=None, plastic=False):
        """Simulate a batch of samples.

        This method will take care that each sample in the batch is properly
        simulated (as specified in the config) as well as evaluating the batch
        simulation.

        Args:
            batch: A list of Samples.
            ident: A batch identifier, i.e.: 'training', 'validation' or
                'test'.
            epoch: The current simulation epoch. This is an important
                information for certain logging mechanism.
            plastic: Whether to turn on synaptic plasticity during simulation
                or not.

        Returns:
            error, i.e. 1 - accuracy.
        """
        network = self.network

        start_time = time.time()

        if plastic:
            self._eq_state.plastic = True

        repetitions = [] # Number of repetitions per sample
        output_rates = []
        true_labels = []
        for ind, sample in enumerate(batch):
            if ind > 0 and ind % config.feedback_interval == 0:
                epoch_msg = ' in current epoch (%d)' % epoch \
                    if epoch is not None else ''
                msg = ('%d %s sample/s have been presented' % (ind, ident) \
                       + '%s so far.' % (epoch_msg))
                logger.info(msg)

            inputs = sample.inputs

            # Decay or resting period.
            network.set_input_rates(np.zeros(inputs.size))
            self.run(config.resting_period * b2.ms)

            # Present sample.
            self._eq_state.set_output(sample.label)

            network.set_input_rates(inputs)
            self.run(config.presentation_period * b2.ms)

            num_output_spikes = np.sum(network.output_spike_counts())

            # Repeat presentation if not enough output spikes have been
            # generated.
            num_repetitions = 0

            fr_range = config.input_fr_max - config.input_fr_min
            while num_output_spikes < config.min_output_spikes:
                num_repetitions += 1

                fr_range_prev = fr_range
                fr_range += config.input_fr_increment
                fr_factor = fr_range / fr_range_prev
                inputs = (inputs - config.input_fr_min) * fr_factor \
                         + config.input_fr_min

                network.set_input_rates(inputs, reset_fr=False)
                self.run(config.presentation_period * b2.ms)

                num_output_spikes = np.sum(network.output_spike_counts())

                if num_repetitions >= config.max_num_repetitions:
                    logger.warning('Maximum number of repetitions reached.')
                    break
            repetitions.append(num_repetitions)

            self._eq_state.unset_output()

            output_rates.append( \
                network.exc_firing_rates(network.num_layers - 1))
            true_labels.append(sample.label)

        if plastic:
            self._eq_state.plastic = False


        logger.info('### Average simulation time per %s sample: %f sec.' \
                    % (ident, (time.time() - start_time) / len(batch)))

        repetitions = np.array(repetitions)
        if np.max(repetitions) > 0:
            logger.debug('Some samples haven\'t produced enough output ' \
                         + 'during the first presentation:')
            logger.debug(' - Total # of repetitions in batch: %d' \
                         % np.sum(repetitions))
            logger.debug(' - Mean # of repetitions in batch: %f' \
                         % np.mean(repetitions))
            logger.debug(' - Std dev of # of repetitions in batch: %f' \
                         % np.std(repetitions))
            logger.debug(' - Median # of repetitions in batch: %f' \
                         % np.median(repetitions))

        ## Evaluate batch error.
        store_eval = True if ident == 'test' \
            else config.store_output_eval_progress
        err = evaluation.evaluate_output_rates(output_rates, true_labels, \
            network.data.output_size, ident, epoch=epoch, store=store_eval)

        return err

    def run(self, duration):
        """Wrapper for the Brian2 run method.

        This method will notify observers about the new simulation time, as
        soon as the simulation is complete.

        Args:
            duration: How long the current network state shall be simulated.

        Returns:
        """
        b2_net = self.network.network
        # We could let Brian send us reports about the current status by
        # passing a callable function
        # my_report(elapsed, complete, tstart, duration)
        # elapsed: Real time elapsed since call of run.
        # complete: Relative simulation progress (0 to 1).
        # tstart: Simulation time before call to run.
        # duration: With what duration was run called.
        #
        # However, as we rather have short calls to the method run, we can wrap
        # reporting completely outside of Brian.
        b2_net.run(duration, profile=config.simulation_profiling)

        # Notify observers.
        self.update_observers(self.__class__.__name__,
                              curr_sim_time=np.float32(b2.defaultclock.t_))

        # Profiling information is overwritten for every new call to run.
        if config.simulation_profiling:
            for name, time in b2_net.profiling_info:
                self._profiler[name] += time / b2.second

if __name__ == '__main__':
    pass


