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
@title           :util/snapshots.py
@author          :ch
@contact         :christian@ini.ethz.ch
@created         :04/19/2017
@version         :1.0
@python_version  :3.5.2

Store/restore a snapshot of the simulation.
"""

import configuration as config

import os
import shutil
from datetime import datetime
import _pickle as pickle
import types
import importlib

import logging
logger = logging.getLogger(config.logging_name)

def store(config, network, pattern, induction, eq_state, epoch=None):
    """Store a snapshot of the simulation in the folder 'snapshots'.

    Snapshots are stored via pickle as python dictionaries.

    Args:
        config: The current configuration module.
        network: Instance of the class NetworkModel.
        pattern: Instance of the class PatternGeneration.
        induction: Instance of the class PatternInduction.
        eq_state: Instance of the class EqStateVars.
        epoch: The epoch, where the snapshot has been saved. If None, it is
            assumed to be a final snapshot, that can't be restored.

    Returns:
    """
    # FIXME pickle cannot serialize the objects we want to dump.
    err_msg = 'Snapshot creation is deactivated due to known issues.'
    logger.warning(err_msg)
    return

    if os.path.isdir(config.snapshot_dir):
        shutil.rmtree(config.snapshot_dir)
    else:
        os.mkdir(config.snapshot_dir)

    restorable = False

    # config is a module and can therefore not be written to a file via pickle.
    # Hence, we extract the variables from config.
    dumpable_config = dict()
    ignored_attrs = []
    for attr_name in dir(config):
        attr = getattr(config, attr_name)
        if attr_name.startswith('_') or isinstance(attr, types.ModuleType) \
                                     or isinstance(attr, type):
            ignored_attrs.append(attr_name)
            continue

        dumpable_config[attr_name] = attr

    logger.debug('The following attributes have been ignored when dumping ' \
                 + 'the configuration to a snapshot: %s' \
                 % (str(ignored_attrs)))

    cur_time = datetime.now()
    snapshot_name = 'snapshot_' + cur_time.strftime('%Y-%m-%d_%H-%M-%S')
    if epoch is None:
        snapshot_name += '_final'
    else:
        restorable = True
        snapshot_name += '_epoch_%d' % epoch
    snapshot_name += '.p'

    snapshot_name = os.path.join(config.snapshot_dir, snapshot_name)

    snapshot = dict()
    snapshot['timestamp'] = cur_time.timestamp()
    snapshot['restorable'] = restorable
    snapshot['config'] = dumpable_config
    snapshot['network'] = network
    snapshot['pattern'] = pattern
    snapshot['induction'] = induction
    snapshot['eq_state'] = eq_state
    snapshot['epoch'] = epoch

    with open(snapshot_name, 'wb') as f:
        pickle.dump(snapshot, f)
    logger.info('### Snapshot has been stored in %s.' % snapshot_name)

def restore(filename):
    """Load a snapshot from file and return the arguments from method 'store'.

    Args:
        filename: Path + name of pickle snapshot.

    Returns:
        See arguments of method 'store'.
        config, network, pattern, induction, eq_state, epoch
    """
    if not os.path.isfile(filename):
        err_msg = 'Could not resore snapshot from file %s.' % filename
        logger.critical(err_msg)
        raise Exception(err_msg)

    with open(filename, 'rb') as f:
        snapshot = dump.load(f)

    if not snapshot['restorable']:
        err_msg = 'Cannot restore a final snapshot (%s).' % filename
        logger.critical(err_msg)
        raise Exception(err_msg)

    # The config has not been dumped as a module, so we need to reset the
    # module variables.
    for attr_name, attr in snapshot['config'].items():
        setattr(config, attr_name, attr)

    # Make sure, the correct equations are known.
    config._equation_module = importlib.import_module('equations.' + \
                                                      config.equation_module)

    return config, snapshot['network'], snapshot['pattern'], \
           snapshot['induction'], snapshot['eq_state'], snapshot['epoch']

if __name__ == '__main__':
    pass


