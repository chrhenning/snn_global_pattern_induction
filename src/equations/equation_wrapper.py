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
@title           :equations/equation_wrapper.py
@author          :ch
@contact         :christian@ini.ethz.ch
@created         :04/12/2017
@version         :1.0
@python_version  :3.5.2

This wrapper is a template to specify equations that define neurons and
synapses. Note, do not modify this file.

To specify a costumized set of equations, one has to copy this file and modify
its attributes to match the desired outcome.

One has to adhere to the following rules, when costumizing a copy of this file.
- DO NOT DEFINE EXTERNAL VARIABLES IN THE NAMESPACE OF THIS FILE. Instead,
  variables should be either defined within the equation strings or within the
  associated namespace dictionary.
- All variables defined in this wrapper have to be specified. However, they may
  be empty.
- All learning rules must be masked by the boolean 'plastic'. See below for
  details.
- All objects have an extra dictionary 'initialization'. That field can be used
  to initialize states (e.g., synaptic weights) with certain variables. The
  values must already hold the correct unit.
  (Note, all the other variable names in this module are based on parameters
  of the Brian classes NeuronGroup resp. Synapses.)
  Note, the initialization must match the dimension (in each layer) for
  non-shared states.
  Alternatively, one can pass a function or lambda expression, that
  receives the dimensions as argument (tuple). E.g., a function from the module
  'weight_initialization'.
  Even for sparse connections, a full matrix has to be passed. Non-existing
  synapses will ignore their associated value during initialization.
  When initializing synaptic weights, the weight_initialization method chosen
  in config is ignored.
  (Matrices must be of type numpy.ndarray).
- Besides 'plastic', NeuronGroups and Synapses may specify other shared
  parameters in there equations, that will be updated by the simulator:
      - 'plastic': This boolean is only True during the training phase. It is
        important that all learning rules (in Synapses and/or NeuronGroup) are
        masked using this parameter, that can en-/disable synaptic plasticity.
      - 'layer': The index of the current layer (0 corresponds to input).
        For Synapses, the index will be the layer index of the postsynaptic
        layer.
      - 'output': The index of the currently presented output class. Note, use
        this parameter with care. It has only a sensible (defined) value,
        during the presentation of a sample and while training. Therefore, only
        use this variable in combination with the parameters 'plastic' and
        'presenting'.
      - 'presenting': Boolean, that is true if a sample is presented to the
        network (i.e., input is present).
      - 'reward': This is true, if a reward is present.
        NOTE, currently not implemented (always False).
      - 'patmod': Pattern modulation of the update, which modifies the
        postsynaptic potential. This variable is used when chosing the
        induction method 'impact_modulation'. See config for details.
   These parameters can be declared on Synapses and NeuronGroups. Note, that a
   Synapse may not have a parameter, that has been already declared at a
   postsynapric neuron. (I.e., do not declare a parameter at a Synapse if it
   has been declared at the postsynaptic NeuronGroup.)

To use the equations in the simulation, the following steps must be taken.
1. Copy this file (within this subfolder) and specify the dynamics.
2. Assign the module name to the variable with name 'equation_module' in the
   configuration file.
"""

########################################
### Neuron Model
########################################
# In the following, the dynamics of neurons are defined.

# Every may exists of excitatory and inhibitory neurons. One may either specify
# the dynamics of all excitatory resp. inhibitory neurons in the network or
# provide layer-wise dynamics.
# In case, one wishes to specify layer-wise dynamics, variables belonging
# to inhibitory resp. excitatory neurons have to be arrays of size: number of
# hidden layers + 1 (one output layer).
# Neural dynamics of the input layer can't be specified.
# NOTE, the membrane voltage must be called 'v' (unit: volt).

## Specify dynamics of excitatory neurons.
ne_model = ''
ne_method = ('linear', 'euler', 'heun')
ne_threshold = None
ne_reset = None
ne_refractory = False
ne_namespace = None
ne_initialization = None

## Specify dynamics of inhibitory neurons.
ni_model = ''
ni_method = ('linear', 'euler', 'heun')
ni_threshold = None
ni_reset = None
ni_refractory = False
ni_namespace = None
ni_initialization = None

########################################
### Synapse Model
########################################
# In the following, the dynamics of synapses are defined.

# The following types of synapses exist:
# - excitatory-excitatory preojections ('ee'): Synapses, that connect exc.
#   neurons in layer l with exc. neurons in layer l+1.
# - excitatory-inhibitory synapses ('ei'): Synapses from exc. neurons to inh.
#   neurons within a layer.
# - inhibitory-excitatory synapses ('ie'): Synapses from inh. neurons to exc.
#   neurons within a layer.
#
# The synaptic dynamics can again be specified layer-wise by defining an array
# of size: number of hidden layers + 1.
# In the case of ee connections, the first model in this array would define the
# connections from the input layer to the first hidden layer.
#
# NOTE, the synaptic weight must be called 'w' (arbitrary unit).
# NOTE, don't forget to integrate the flag 'plastic' for all plastic synapses,
# which can turn on/off synaptic plasticity during training/test phase.
# Example applications of the flag plastic, when updating the synaptic weight:
#   'w = new_w * plastic + w * (1-plastic)'
#   'w += update_w * plastic'

## Dynamics of excitatory-excitatory projections.
ee_model = 'plastic : boolean (shared, constant)'
ee_method = ('linear', 'euler', 'heun')
ee_on_pre = None
ee_on_post = None
ee_delay = None
ee_namespace = None
ee_initialization = None

## Dynamics of excitatory-inhibitory synapses.
ei_model = 'plastic : boolean (shared, constant)'
ei_method = ('linear', 'euler', 'heun')
ei_on_pre = None
ei_on_post = None
ei_delay = None
ei_namespace = None
ei_initialization = None

## Dynamics of inhibitory-excitatory synapses.
ie_model = 'plastic : boolean (shared, constant)'
ie_method = ('linear', 'euler', 'heun')
ie_on_pre = None
ie_on_post = None
ie_delay = None
ie_namespace = None
ie_initialization = None

if __name__ == '__main__':
    pass

