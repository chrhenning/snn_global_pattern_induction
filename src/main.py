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
@title           :main.py
@author          :ch
@contact         :christian@ini.ethz.ch
@created         :03/15/2017
@version         :1.0
@python_version  :3.5.2

This module should be executed to simulate the network. It determines the
workflow of the SNN-SPD simulation.

The module contains the main method of this package, that is aimed to simulate
an SNN-SPD (a Spiking Neural Network with Stimulated Pattern Development).
"""

import configuration as config
from util.config_exception import ConfigException
from data.mnist_data import MNISTData
from data.seven_segment_data import SevenSegmentData
from pattern_generation import PatternGeneration
from pattern_induction import PatternInduction
from network_model import NetworkModel
from visualization import draw_network
from visualization import draw_patterns
from eq_state_vars import EqStateVars
from util import snapshots
from util import utils
from recordings import Recordings
from simulation import Simulation

import itertools
import numpy as np
import random
from brian2 import seed as bseed
import multiprocessing
import time
import os
import shutil
import brian2 as b2
import importlib
import matplotlib.pyplot as plt

if __name__ == '__main__':
    start_time = time.time()

    logger = utils.config_logger(config.logging_name, config.logging_logfile,
                                 config.file_loglevel, config.console_loglevel)
    logger.info('### Stimulated Pattern Development in SNN ###')

    # Import the chosen equation module.
    config._equation_module = importlib.import_module('equations.' + \
                                                      config.equation_module)

    # Make all random processes predictable.
    np.random.seed(config.random_seed)
    random.seed(config.random_seed)
    bseed(config.random_seed) # See brian docu for explanation.

    # Determine maximum number of threads to use.
    if config.num_threads is None:
        config.num_threads = multiprocessing.cpu_count()

    # Read the chosen dataset.
    logger.info('### Data preparation ...')
    if config.dataset == 'mnist':
        data = MNISTData()
    elif config.dataset == '7segment':
        data = SevenSegmentData()
    else:
        raise ConfigException('The chosen dataset \'%s\' is unknown. Please ' \
                              % config.dataset + 'reconsider the ''dataset''' \
                              + ' option of the configuration file.')
    logger.info('### Data preparation ... Done')


    # Assemble the network.
    logger.info('### Building Network ...')
    network = NetworkModel(data)
    logger.info('### Building Network ... Done')

    # Visualize just assembled network.
    if config.plot_network or config.save_network_plot:
        logger.info('### Generating Network Plot ...')
        draw_network.draw_network(network)
        logger.info('### Generating Network Plot ... Done')

    if config.induce_patterns:
        # Generate and induce patterns.
        logger.info('### Patterns are induced to the network ...')
        pattern = PatternGeneration(network)

        # Visualize patterns.
        if config.plot_patterns or config.save_pattern_plot:
            logger.info('### Generating Pattern Plots ...')
            draw_patterns.draw_patterns(network, pattern)
            logger.info('### Generating Pattern Plots ... Done')

        if config.invert_patterns:
            pattern.invert_patterns()
        pattern.apply_layer_wise_decay()
        pattern.scale_strength()

        induction = PatternInduction(network, pattern)
        logger.info('### Patterns are induced to the network ... Done')
    else:
        pattern = None
        induction = None

    logger.info('### Setup Time: %f sec' \
                % (time.time() - start_time))

    # TODO do not delete when restoring snapshots.
    ## Ensure, that there are no evaluation results from previous simulations.
    if os.path.isdir(config.eval_dir):
        shutil.rmtree(config.eval_dir)
    ## Ensure, that previous recordings are erased.
    if os.path.isdir(config.recording_dir):
        shutil.rmtree(config.recording_dir)

    ## Set up recordings.
    recordings = Recordings(network)

    logger.info('### Simulation ...')

    simulation = Simulation()
    simulation.network = network

    eq_state = EqStateVars()
    eq_state.set_defaults()

    # Set simulation time step.
    b2.defaultclock.dt = config.simulation_timestep * b2.ms

    # Setup all equation. I.e., generate code to solve neural dynamics.
    simulation.run(0*b2.ms)
    # Supress info messages from now on, if user hasn't specified the
    # integration method.
    b2.utils.logger.BrianLogger.suppress_name('method_choice')
    # Probably we also don't wanna show codegen warnings anymore, as they are
    # sometimes not useful.
    b2.BrianLogger.suppress_hierarchy('brian2.codegen')

    train = data.train
    test = data.test
    if data.val is None:
        validation = test
    else:
        validation = data.val

    to_abs_size = lambda x, s : x if isinstance(x, int) else round(x*s)

    train_batch_size = to_abs_size(config.train_batch_size, len(train))
    val_batch_size = to_abs_size(config.val_batch_size, len(validation))
    logger.info('### Using training batch size of %d.' % (train_batch_size))
    logger.info('### Using validation batch size of %d.' % (val_batch_size))

    train_batch_gen = utils.random_shuffle_loop(train)
    val_batch_gen = utils.random_shuffle_loop(validation)

    # The validation batch is chosen once and then hold fix for the whole
    # simulation, such that validation accuracies are comparable between
    # trials.
    val_batch = list(itertools.islice(val_batch_gen, val_batch_size))

    epoch = 1
    while True:
        epoch_start_time = time.time()
        logger.info('### Starting simulation epoch %d.' % epoch)

        # Store snapshot.
        if epoch > 1 and (epoch+1) % config.snapshot_every == 0:
            snapshots.store(config, network, pattern, induction, eq_state,
                            epoch)

        train_batch = list(itertools.islice(train_batch_gen,
                                            train_batch_size))

        ## Present the whole training batch to the network.
        logger.info('Processing training batch ...')
        train_err = simulation.simulate_batch(train_batch, 'training',
                                              epoch=epoch, plastic=True)
        logger.info('Processing training batch ... Done')

        ## Evaluate current network state.
        if epoch % config.eval_interval == 0:

            # Present the validation batch to the network
            logger.info('Validating network ...')
            val_err = simulation.simulate_batch(val_batch, 'validation',
                                                epoch=epoch, plastic=False)
            logger.info('Validating network ... Done')

            logger.info('### Generalization Error in epoch %d: %f' \
                        % (epoch, abs(train_err - val_err)))

            if val_err <= config.validation_error:
                logger.info('### Convergence condition satisfied: validation' \
                            + ' error %f (<= %f).' \
                            % (val_err, config.validation_error))
                break

        if config.num_epochs is not None and epoch == config.num_epochs:
            logger.info('### Maximum number of epochs is reached.')
            break

        if pattern is not None:
            pattern.decay_influence()
        logger.info('### Epoch %d has been processed in: %f sec' \
                    % (epoch, time.time() - epoch_start_time))

        epoch += 1

    # Generate final snapshot.
    snapshots.store(config, network, pattern, induction, eq_state)

    # Present the whole test set to the network
    # FIXME uncomment
    #logger.info('Presenting complete test set to the learned network ...')
    #test_err = simulation.simulate_batch(test, 'test', plastic=False)
    #logger.info('The simulation finishes with an overall test error of: %f' \
    #            % test_err)
    logger.info('Presenting complete test set to the learned network ... Done')

    logger.info('### Simulation ... Done')

    logger.info('### Save Recordings ...')
    recordings.store_recordings()
    logger.info('### Save Recordings ... Done')

    ## Log profiling infos, if desired.
    if config.simulation_profiling:
        logger.debug('Profiling summary:')
        max_name = None
        max_time = -1
        for name, t in simulation.profiler.items():
            if t > max_time:
                max_name = name
                max_time = t
            logger.debug(' - %s: %f sec' % (name, t))
        if max_name is not None:
            logger.debug('Computationally most demanding component: ' + \
                         '%s (%f sec).' % (max_name, max_time))

    end_time = time.time()
    logger.info('### Overall Runtime: %f sec' \
                % (end_time - start_time))

