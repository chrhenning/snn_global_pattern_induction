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
@title           :equations/Simple_CurrLIF_and_STDP.py
@author          :ch
@contact         :christian@ini.ethz.ch
@created         :05/07/2017
@version         :1.0
@python_version  :3.5.2

Current-based LIF neurons with simple STDP rules.

The stimulation of neurons happens due to synaptic currents, that are scaled by
the synaptic weight and modulated by an alpha function as described in:
    Gerstner, W. & Kistler, W. (2002). Spiking Neuron Models: Single Neurons,
    Populations, Plasticity. Cambridge University Press
"""

import brian2 as b2
import numpy as np

########################################
### Neuron Model
########################################
## Specify dynamics of excitatory neurons.
# Simple LIF neurons.
ne_model = '''
    dv/dt = (-v + R*I) / tau : volt
    I = I_ee + I_ie          : amp
    I_ee                     : amp
    I_ie                     : amp
'''
ne_method = ('linear', 'euler', 'heun')
ne_threshold = 'v > v_thresh'
ne_reset = 'v = v_reset'
ne_refractory = 't_refrac'
ne_namespace = {
    'R': 1 * b2.ohm,        # Input resistence
    'tau': 10 * b2.ms,      # Time constant
    'v_thresh': 1 * b2.volt,# Threshold
    'v_reset': 0 * b2.volt, # Reset Value
    't_refrac': 20 * b2.ms  # Refractory Period
}
ne_initialization = {
    'v': lambda size: np.ones((size)) * 0 * b2.volt
}

## Specify dynamics of inhibitory neurons.
# Simple LIF neurons.
ni_model = '''
    dv/dt = (-v + R*I) / tau : volt
    I = I_ei                 : amp
    I_ei                     : amp
'''
ni_method = ('linear', 'euler', 'heun')
ni_threshold = 'v > v_thresh'
ni_reset = 'v = v_reset'
ni_refractory = 't_refrac'
ni_namespace = {
    'R': 1 * b2.ohm,        # Input resistence
    'tau': 10 * b2.ms,      # Time constant
    'v_thresh': 1 * b2.volt,# Threshold
    'v_reset': 0 * b2.volt, # Reset Value
    't_refrac': 20 * b2.ms  # Refractory Period
}
ni_initialization = {
    'v': lambda size: np.ones((size)) * 0 * b2.volt
}

########################################
### Synapse Model
########################################
# Simple STDP rules based on a pre- and postsynaptic spike trace.
# I.e., decrease on presynaptice spike if there has been a recent postsynaptic
# spike and increase on postsynaptic spike if there has been a recent
# presynaptic spike.
# The weight influences the postsynaptic neuron as it scales the injected
# current. The current is modulated via an alpha function, that decays
# exponentially after a presynaptic spike occurred.
# Variable meanings:
# - 'so': "Spike occurred", set to one after the first spike occurred.
# - 't_pr': The time, when the last presynaptic spike occurred.
# Note, only the last presynaptic spike has an influence to the current
# injection.

## Dynamics of excitatory-excitatory projections.
# Exc.-exc. connections implement the variable 'patmod', that modulates the
# synaptic influence if the appropriate pattern induction method was chosen.
# Otherwise it has not impact.
ee_model = '''
    w                                           : 1
    I_ee_post = (w+patmod) * alpha * amp        : amp (summed)
    alpha = alph * s_alph * exp(1-s_alph) * so  : 1
    s_alph = (t - t_pr) / tau_alph              : 1
    so                                          : 1
    t_pr                                        : second
    dtr_pr/dt = -tr_pr / tau_pr                 : 1 (event-driven)
    dtr_po/dt = -tr_po / tau_po                 : 1 (event-driven)
    plastic                                     : boolean (shared, constant)
    patmod                                      : 1
'''
ee_method = ('linear', 'euler', 'heun')
ee_on_pre = '''
    t_pr = t
    so = 1
    tr_pr = tr_pr + 1
    w = clip(w - (a_po*tr_po) * plastic, 0, 1)
'''
ee_on_post = '''
    tr_po = tr_po + 1
    w = clip(w + (a_pr*tr_pr) * plastic, 0, 1)
'''
ee_delay = None
ee_namespace = {
    'tau_pr': 10 * b2.ms,   # Time constant for presynaptic spike trace
    'tau_po': 10 * b2.ms,   # Time constant for presynaptic spike trace
    'a_pr' : 0.001,           # Influence of presynaptic trace on weight
    'a_po' : 0.001,           # Influence of presynaptic trace on weight
    'alph' : 4,           # Scale influence of synaptic current injection
    'tau_alph' : 10 * b2.ms # Time constant for decay of synaptic current
}
ee_initialization = None

## Dynamics of excitatory-inhibitory synapses.
ei_model = '''
    w                                           : 1
    I_ei_post = alpha * w * amp                 : amp (summed)
    alpha = alph * s_alph * exp(1-s_alph) * so  : 1
    s_alph = (t - t_pr) / tau_alph              : 1
    so                                          : 1
    t_pr                                        : second
    dtr_pr/dt = -tr_pr / tau_pr                 : 1 (event-driven)
    dtr_po/dt = -tr_po / tau_po                 : 1 (event-driven)
    plastic                                     : boolean (shared, constant)
'''
ei_method = ('linear', 'euler', 'heun')
ei_on_pre = '''
    t_pr = t
    so = 1
    tr_pr = tr_pr + 1
    w = clip(w - (a_po*tr_po) * plastic, 0, 1)
'''
ei_on_post = '''
    tr_po = tr_po + 1
    w = clip(w + (a_pr*tr_pr) * plastic, 0, 1)
'''
ei_delay = None
ei_namespace = {
    'tau_pr': 10 * b2.ms,   # Time constant for presynaptic spike trace
    'tau_po': 10 * b2.ms,   # Time constant for presynaptic spike trace
    'a_pr' : 0.01,           # Influence of presynaptic trace on weight
    'a_po' : 0.01,           # Influence of presynaptic trace on weight
    'alph' : 10,           # Scale influence of synaptic current injection
    'tau_alph' : 10 * b2.ms # Time constant for decay of synaptic current
}
ei_initialization = None

## Dynamics of inhibitory-excitatory synapses.
ie_model = '''
    w                                           : 1
    I_ie_post = alpha * (-(w-patmod)) * amp              : amp (summed)
    alpha = alph * s_alph * exp(1-s_alph) * so  : 1
    s_alph = (t - t_pr) / tau_alph              : 1
    so                                          : 1
    t_pr                                        : second
    dtr_pr/dt = -tr_pr / tau_pr                 : 1 (event-driven)
    dtr_po/dt = -tr_po / tau_po                 : 1 (event-driven)
    plastic                                     : boolean (shared, constant)
    patmod                                      : 1
'''
ie_method = ('linear', 'euler', 'heun')
ie_on_pre = '''
    t_pr = t
    so = 1
    tr_pr = tr_pr + 1
    w = clip(w - (a_po*tr_po) * plastic, 0, 1)
'''
ie_on_post = '''
    tr_po = tr_po + 1
    w = clip(w + (a_pr*tr_pr) * plastic, 0, 1)
'''
ie_delay = None
ie_namespace = {
    'tau_pr': 10 * b2.ms,   # Time constant for presynaptic spike trace
    'tau_po': 10 * b2.ms,   # Time constant for presynaptic spike trace
    'a_pr' : 0.01,           # Influence of presynaptic trace on weight
    'a_po' : 0.01,           # Influence of presynaptic trace on weight
    'alph' : 4,           # Scale influence of synaptic current injection
    'tau_alph' : 10 * b2.ms # Time constant for decay of synaptic current
}
ie_initialization = None

if __name__ == '__main__':
    pass


