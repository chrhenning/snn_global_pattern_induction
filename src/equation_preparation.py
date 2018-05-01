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
@title           :equation_preparation.py
@author          :ch
@contact         :christian@ini.ethz.ch
@created         :04/13/2017
@version         :1.0
@python_version  :3.5.2

This module takes the chosen set of equations and prepares them for the usage
of the simulator.

The user has several degrees of freedom, when defining equations in a copy of
the file 'src/equations/equation_wrapper'. Therefore, the equation module
chosen via config.equation_module is prepared, such that all its variables are
easily accessible.
This means mainly, converting single value variables into arrays.
"""

import configuration as config
from util.config_exception import ConfigException

import brian2 as b2

import logging
logger = logging.getLogger(config.logging_name)

def prepare_equations(num_costumizable_layers):
    """Prepare the equations in the chosen equation module such that the
    variables have a uniform interface.

    Args:
        num_costumizable_layers: The number of layers that can be costumized
            (number of hidden layers + 1).

    Returns:
    """
    em = config._equation_module

    num_layers = num_costumizable_layers

    # Check neuron equations.
    _transform_to_array(em, ['ne_model', 'ne_method', 'ne_threshold',
                             'ne_reset', 'ne_refractory', 'ne_namespace',
                             'ne_initialization'], num_layers)

    _transform_to_array(em, ['ni_model', 'ni_method', 'ni_threshold',
                             'ni_reset', 'ni_refractory', 'ni_namespace',
                             'ni_initialization'], num_layers)

    # Check synaptic equations.
    _transform_to_array(em, ['ee_model', 'ee_method', 'ee_on_pre',
                             'ee_on_post', 'ee_delay', 'ee_namespace',
                             'ee_initialization'], num_layers)

    _transform_to_array(em, ['ei_model', 'ei_method', 'ei_on_pre',
                             'ei_on_post', 'ei_delay', 'ei_namespace',
                             'ei_initialization'], num_layers)

    _transform_to_array(em, ['ie_model', 'ie_method', 'ie_on_pre',
                             'ie_on_post', 'ie_delay', 'ie_namespace',
                             'ie_initialization'], num_layers)

    # Make sure states have been chosen correctly.
    _check_state_vars(em, ['ee_model', 'ee_on_pre', 'ee_on_post', \
                           'ei_model', 'ei_on_pre', 'ei_on_post', \
                           'ie_model', 'ie_on_pre', 'ie_on_post'])

def _transform_to_array(eq_module, variables, size):
    """Transform the variables, specified by a set of strings, into arrays if
    they are single value.

    Args:
        eq_module: A module, that contains global variables, that can be
            accessed.
        variables: A set of strings, specifying variables in the module
            eq_module.
        size: The size of the arrays to generate.

    Returns:
    """
    for var_name in variables:
        assert(hasattr(eq_module, var_name))
        var = getattr(eq_module, var_name)
        if isinstance(var, list):
            if len(var) != size:
                raise ConfigException('Wrongly configured equations. The ' \
                                      + 'attribute %s has not the correct ' \
                                      % (var_name) + 'length.')
        else:
            setattr(eq_module, var_name, [var] * size)

def _check_state_vars(eq_module, variables):
    """Some state variables need extra treatment. This can be done in this
    method.

    - 'patmod': if 'impact_modulation' has been chosen as induction method,
      then this variable is required to effect synaptic dynamics in any way. If
      it is not included in any of the synaptic equations, a new equation is
      added to the presynaptic dynamics of exc.-exc. synapses.

    Args:
        eq_module: A module, that contains global variables, that can be
            accessed.
        variables: A set of strings, specifying variables in the module
            eq_module.

    Returns:
    """
    if config.induce_patterns and \
       config.pattern_induction_method == 'impact_modulation':
        # Check patmod.
        # No complex checking has been implemented. If check not possible, we
        # just abort it. It is later checked wether 'patmod' is part of at
        # least one synaptic state. If not, the program is cancelled.
        abort_check = False

        patmod_considered = False
        for var_name in variables:
            var = getattr(eq_module, var_name)
            for var_elem in var:
                if isinstance(var_elem, str):
                    if 'patmod' in var_elem:
                        patmod_considered = True
                        break
                elif isinstance(var_elem, (b2.Equations, dict)):
                    abort_check = True
                    break
            if patmod_considered or abort_check:
                break

        # The user has not considered patmod, therefore, it is additively
        # applied to the postsynaptic membrane here.
        if not patmod_considered and not abort_check:
            # Force default to be zero.
            config.impact_modulation_default = 0

            ee_model = getattr(eq_module, 'ee_model')
            ee_on_pre = getattr(eq_module, 'ee_on_pre')

            error_msg = 'Induction method %s requires consideration of ' \
                    % (config.pattern_induction_method) + 'synaptic variable'\
                    + ' \'patmod\'.'

            for i in range(len(ee_model)):
                eq = 'patmod : 1'
                if ee_model[i] is None:
                    ee_model[i] = eq
                elif isinstance(ee_model[i], b2.Equations):
                    raise ConfigException(error_msg)
                else:
                    ee_model[i] += '\n' + eq

            for i in range(len(ee_on_pre)):
                eq = 'v_post += patmod * volt'
                if ee_on_pre[i] is None:
                    ee_on_pre[i] = eq
                elif isinstance(ee_on_pre[i], dict):
                    raise ConfigException(error_msg)
                else:
                    ee_on_pre[i] += '\n' + eq

            setattr(eq_module, 'ee_model', ee_model)
            setattr(eq_module, 'ee_on_pre', ee_on_pre)

            logger.warning('Induction method %s has been chosen, but synaptic'\
                           % (config.pattern_induction_method) + ' variable ' \
                           + '\'patmod\' has not been introduced. Therefore, '\
                           + 'patterns do now increase the postsynpatic ' \
                           + 'potential on presynaptic spikes according to ' \
                           + 'their pattern weight.')

if __name__ == '__main__':
    pass

