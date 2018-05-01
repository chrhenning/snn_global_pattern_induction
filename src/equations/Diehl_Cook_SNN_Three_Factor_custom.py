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
@title           :equations/Diehl_Cook_SNN_Three_Factor_custom.py
@author          :ch
@contact         :christian@ini.ethz.ch
@created         :05/07/2017
@version         :1.0
@python_version  :3.5.2

This module is a copy of the module "Diehl_Cook_SNN_Three_Factor, but with
customized hyperparamters for a specific network structure.
TODO describe network structure

Therefore, we note that the neuronal dynamics are taken from:

'Unsupervised learning of digit recognition using spike-timing-dependent
plasticity', Diehl Peter, Cook Matthew, 2015 (10.3389/fncom.2015.00099)
"""

import brian2 as b2
import numpy as np

import weight_initialization as winit

########################################
### Neuron Model
########################################

## Specify dynamics of excitatory neurons.
# Conductance-based Leaky-Integrate-and-Fire neuron.
ne_model = '''
    dv/dt = ((v_rest_e - v) + (I_synE+I_synI) / nS) / (100*ms)  : volt
    I_synE = ge * nS * (-v)                                     : amp
    I_synI = gi * nS * (-100.*mV - v)                           : amp
    dge/dt = -ge/(1.0*ms)                                       : 1
    dgi/dt = -gi/(2.0*ms)                                       : 1
'''
# They implemented an adaptive threshold to introduce homeostasis.
# This threshold should not change during test phase.
ne_model += '''\n
    plastic                                    : boolean (shared, constant)
    dtheta/dt = (-theta / (tc_theta)) * plastic                 : volt
    dtimer/dt = 100.0                                           : second
'''
ne_method = 'euler'
# Note, that the second term in the threshold equation should not have any
# effect, as the timer runs a 100 times faster. So the refractory period won't
# be over when that condition is reached (no spikes during refractory period).
# Not sure, if this is intentional.
ne_threshold = '(v > (theta - offset + v_thresh_e)) and (timer > refrac_e)'
ne_reset = '''
    v = v_reset_e
    timer = 0*ms
    theta += theta_plus_e * plastic
'''
ne_refractory = 'refrac_e'
ne_namespace = {
    'v_rest_e': -65. * b2.mV,    # Resting potential
    'tc_theta': 1e4 * b2.ms,     # Time constant for adapive threshold var
    'offset': 20.0 * b2.mV,      # Homestasis offset
    'v_thresh_e': -52. * b2.mV,  # Basic firing threshold (no adaption)
    'refrac_e': 5. * b2.ms,      # Refractory period
    'theta_plus_e': 0.05 * b2.mV,# Threshold adaption on spike event
    'v_reset_e': -65. * b2.mV    # Reset value
}
ne_initialization = {
    'theta': lambda size: np.ones((size)) * 20.0*b2.mV,
    'v': lambda size: np.ones((size)) * -65.0*b2.mV
}

## Specify dynamics of inhibitory neurons.
ni_model = '''
    dv/dt = ((v_rest_i - v) + (I_synE+I_synI) / nS) / (10*ms)  : volt
    I_synE = ge * nS * (-v)                                    : amp
    I_synI = gi * nS * (-85.*mV-v)                             : amp
    dge/dt = -ge/(1.0*ms)                                      : 1
    dgi/dt = -gi/(2.0*ms)                                      : 1
'''
ni_method = 'euler'
ni_threshold = 'v > v_thresh_i'
ni_reset = 'v = v_reset_i'
ni_refractory = 'refrac_i'
ni_namespace = {
    'v_rest_i': -60. * b2.mV,   # Resting potential
    'v_thresh_i': -40. * b2.mV, # Threshold
    'refrac_i': 2. * b2.ms,     # Refractory period
    'v_reset_i': -45. * b2.mV   # Reset Value
}
ni_initialization = {
    'v': lambda size: np.ones((size)) * -60.0*b2.mV
}

########################################
### Synapse Model
########################################

## Dynamics of excitatory-excitatory projections.
# Uses parameter 'plastic' of postsynaptic neuron.
ee_model = '''
    w                                      : 1
    post2before                            : 1
    dpre/dt = -pre/(tc_pre_ee)             : 1 (event-driven)
    dpost1/dt = -post1/(tc_post_1_ee)      : 1 (event-driven)
    dpost2/dt = -post2/(tc_post_2_ee)      : 1 (event-driven)
'''
ee_method = 'linear'
# Update on presynaptic spike considers only last postsynaptic spike.
# A presynaptic spike causes an increase in the excitatory connection of the
# postsynaptic neuron.
ee_on_pre = '''
    pre = 1.
    w = clip(w - (nu_ee_pre * post1) * plastic, 0, wmax_ee)
    ge += w
'''
# Update on postsynaptic spike considers last post- and last presynaptic spike.
ee_on_post = '''
    post2before = post2
    w = clip(w + (nu_ee_post * pre * post2before) * plastic, 0, wmax_ee)
    post1 = 1.
    post2 = 1.
'''
ee_delay = None
ee_namespace = {
    'tc_pre_ee': 20*b2.ms,   # Time constant presynaptic eligibility trace
    'tc_post_1_ee': 20*b2.ms,# Time constant post-to-pre eligibility trace
    'tc_post_2_ee': 40*b2.ms,# Time constant post-to-post eligibility trace
    'nu_ee_pre':  0.0001,    # Learning rate on pre
    'nu_ee_post': 0.01,      # Learning rate on post
    'wmax_ee': 1.0           # Maximum excitatory weight
}
ee_initialization = {
    'w': lambda size: winit.uniform(size, wmin=0, wmax=.3)
}

## Dynamics of excitatory-inhibitory synapses.
ei_model = 'w : 1'
ei_method = 'linear'
ei_on_pre = 'ge += w'
ei_on_post = None
ei_delay = None
ei_namespace = None
ei_initialization = {
    'w': lambda size: winit.uniform(size, wmin=0, wmax=.3)
}

## Dynamics of inhibitory-excitatory synapses.
ie_model = 'w : 1'
ie_method = 'linear'
ie_on_pre = 'gi += w'
ie_on_post = None
ie_delay = None
ie_namespace = None
ie_initialization = {
    'w': lambda size: winit.uniform(size, wmin=0, wmax=.3)
}

if __name__ == '__main__':
    pass

