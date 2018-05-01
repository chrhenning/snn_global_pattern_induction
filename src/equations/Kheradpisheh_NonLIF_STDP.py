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
@title           :equations/Kheradpisheh_NonLIF_STDP.py
@author          :ch
@contact         :christian@ini.ethz.ch
@created         :06/28/2017
@version         :1.0
@python_version  :3.5.2

Here we implement a slight variant of the neural dynamics suggested in:
    Kheradpisheh, Saeed Reza and Ganjtabesh, Mohammad and Thorpe, Simon J and
    Masquelier, Timothee (2002). STDP-based spiking deep neural networks for
    object recognition. arXiv

The model implements non-leaky integrate-and-fire neuron, that simply increases
its potential everytime a presynaptic neuron spikes (according to the
connection weight). The potential is reset after every spike.

The model learns via STDP. A weight update is performed everytime a presynaptic
spike occurs after a postsynaptic one (weight decrease) and evertime a
postsynaptic spike occurs after a presynaptic one (weight increase).
"""

import brian2 as b2
import numpy as np

########################################
### Neuron Model
########################################
# Non-leaky integrate and fire neuron.

## Specify dynamics of excitatory neurons.
ne_model = 'v : volt'
    #dv/dt = 0 * volt / second       : volt
ne_method = ('linear', 'euler', 'heun')
ne_threshold = 'v > v_thresh'
ne_reset = 'v = v_reset'
ne_refractory = False
ne_namespace = {
    'v_thresh': 1 * b2.volt,    # Threshold
    'v_reset': 0 * b2.volt      # Reset value
}
ne_initialization = {
    'v': lambda size: np.zeros((size)) * b2.volt
}

## Specify dynamics of inhibitory neurons.
ni_model = 'v : volt'
ni_method = ('linear', 'euler', 'heun')
ni_threshold = 'v > v_thresh'
ni_reset = 'v = v_reset'
ni_refractory = False
ni_namespace = {
    'v_thresh': 1 * b2.volt,    # Threshold
    'v_reset': 0 * b2.volt      # Reset value
}
ni_initialization = {
    'v': lambda size: np.zeros((size)) * b2.volt
}

########################################
### Synapse Model
########################################
# STDP rule with weight updates on spike occurence independent of the actual
# timedelay between post- and pre-synaptic spikes. The weights are kept within
# the range [0,1] implicitly due to the update rule.

## Dynamics of excitatory-excitatory projections.
ee_model = '''
    w                    : 1
    plastic              : boolean (shared, constant)
    pr_occurred          : 1
    po_occurred          : 1
'''
ee_method = ('linear', 'euler', 'heun')
ee_on_pre = '''
    w = w + plastic * po_occurred * a_minus * w * (1-w)
    v_post = v_post + w * volt
    pr_occurred = 1
    po_occurred = 0
'''
ee_on_post = '''
    w = w + plastic * pr_occurred * a_plus * w * (1-w)
    po_occurred = 1
    pr_occurred = 0
'''
ee_delay = None
ee_namespace = {
    'a_minus': -0.001,  # Scale post-before-pre weight update
    'a_plus': 0.001     # Scale pre-before-post weight update
}
ee_initialization = {
    'po_occurred': 0,   # last occurring spike has been a post-syn. spike
    'pr_occurred': 0    # last occurring spike has been a pre-syn. spike
}

## Dynamics of excitatory-inhibitory synapses.
ei_model = '''
    w                   : 1
    plastic             : boolean (shared, constant)
    pr_occurred          : 1
    po_occurred          : 1
'''
ei_method = ('linear', 'euler', 'heun')
ei_on_pre = '''
    w = w + plastic * po_occurred * a_minus * w * (1-w)
    v_post = v_post + w * volt
    pr_occurred = 1
    po_occurred = 0
'''
ei_on_post = '''
    w = w + plastic * pr_occurred * a_plus * w * (1-w)
    po_occurred = 1
    pr_occurred = 0
'''
ei_delay = None
ei_namespace = {
    'a_minus': -0.001,  # Scale post-before-pre weight update
    'a_plus': 0.001     # Scale pre-before-post weight update
}
ei_initialization = {
    'po_occurred': 0,   # last occurring spike has been a post-syn. spike
    'pr_occurred': 0    # last occurring spike has been a pre-syn. spike
}

## Dynamics of inhibitory-excitatory synapses.
ie_model = '''
    w                   : 1
    plastic             : boolean (shared, constant)
    pr_occurred          : 1
    po_occurred          : 1
'''
ie_method = ('linear', 'euler', 'heun')
ie_on_pre = '''
    w = w + plastic * po_occurred * a_minus * w * (1-w)
    v_post = v_post - w * volt
    pr_occurred = 1
    po_occurred = 0
'''
ie_on_post = '''
    w = w + plastic * pr_occurred * a_plus * w * (1-w)
    po_occurred = 1
    pr_occurred = 0
'''
ie_delay = None
ie_namespace = {
    'a_minus': -0.001,  # Scale post-before-pre weight update
    'a_plus': 0.001     # Scale pre-before-post weight update
}
ie_initialization = {
    'po_occurred': 0,   # last occurring spike has been a post-syn. spike
    'pr_occurred': 0    # last occurring spike has been a pre-syn. spike
}

if __name__ == '__main__':
    pass

