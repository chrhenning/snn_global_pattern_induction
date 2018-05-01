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
@title           :eq_state_vars.py
@author          :ch
@contact         :christian@ini.ethz.ch
@created         :04/18/2017
@version         :1.0
@python_version  :3.5.2

Some observable external state variables that can be used in equations.

When defining a set of neural and synaptic equations in a copy of the equation
wrapper, some special state variables are allowed or required (such as plastic)
to model more expressive dynamics. Such state variables are the attributes of
this class. Modules that observe this class get notified when the state
variables undergo changes. For instance, the class NetworkModel should adapt
these value of these variables in all Synapses resp. NeuronGroups that are
utilizing them.
"""

import configuration as config

from pypatterns.observable import Observable
from pypatterns.singleton import Singleton

import logging
logger = logging.getLogger(config.logging_name)

class EqStateVars(Observable, metaclass=Singleton):
    """Class that holds a set of state variables and defines appropriate getter
    and setter methods for those. Observers are notified on changes given the
    name of the changed attribute and its old and new value.

    Attributes:
        plastic: Defines, whether the current phase is a training phase or not
            (Synpases should only be plastic during training).
        output: The index of the currently presented output class. Undefined if
            nothing is presented.
        presenting: Whether a sample is presented to the network or not.
        reward: Whether an reward is active or not.
        patmod: 'Pattern Modulation'. A special synaptic variable that shall
            modulate the influence of a synapse on its postsynaptic neuron
            according to the pattern.
    """
    def __init__(self):
        """Set default values.

        Args:

        Returns:
        """
        super().__init__()

        self._plastic = False
        self._output = 0
        self._presenting = False
        self._reward = False
        self._patmod = config.impact_modulation_default

    def set_defaults(self):
        """This method should be called once before the simulation will be
        started but after all observers have been registered.

        It sets all attributes to default values.
        """
        self.plastic = False
        self.unset_output()
        self.reward = False
        self.patmod = config.impact_modulation_default

    @property
    def plastic(self):
        """Getter for attribute plastic.

        Args:

        Returns:
            Whether learning is currently enabled or not.
        """
        return self._plastic

    @plastic.setter
    def plastic(self, value):
        """Setter for attribute plastic.

        Args:
            value: The plasticity state of the network.

        Returns:
        """
        self._custom_setter('plastic', value)

    @property
    def output(self):
        """Getter for attribute output.

        Args:

        Returns:
            The currently presented output class if presenting is True, else
            undefined.
        """
        return self._output


    @property
    def presenting(self):
        """Getter for attribute presenting.

        Args:

        Returns:
            Whether a sample is currently presented to the network.
        """
        return self._presenting

    def set_output(self, iclass):
        """Define the currently presented output class.

         This method sets the attribute output to iclass and the attribute
         presenting to True.

         NOTE, the attribute 'presenting' is always updated after the attribute
         'output' has been updated.

         Args:
             iclass: Index of presented output class.

        Returns:
        """
        self._custom_setter('output', iclass)
        self._custom_setter('presenting', True)

    def unset_output(self):
        """A call to this method says that no output is anymore presented to
        the network, i.e., the attribute presenting is set to False.

        Args:

        Returns:
        """
        self._custom_setter('presenting', False)

    @property
    def reward(self):
        """Getter for attribute reward.

        Args:

        Returns:
            Whether a reward is currently active.
        """
        return self._reward

    @reward.setter
    def reward(self, value):
        """Setter for attribute reward.

        Args:
            value: The reward state of the network.

        Returns:
        """
        self._custom_setter('reward', value)

    @property
    def patmod(self):
        """Getter for attribute patmod.

        Args:

        Returns:
            Is pattern modulation is active, then the pattern is returned,
            otherwise the standard value of the variable.
        """
        return self._patmod

    @patmod.setter
    def patmod(self, value):
        """Setter for attribute patmod.

        Args:
            value: If a single value is given, then this should be applied as
            the standard value to all synapses implementing the variable.
            Otherwise, a list of patterns per layer is given. Each layer
            contains is a vector, that defines a pattern value for all
            postsynaptic neurons (i.e., all synapses with the same postsynaptic
            neuron get the same value for patmod).

        Returns:
        """
        self._custom_setter('patmod', value)


    def _custom_setter(self, attr_name, value):
        oval = getattr(self, attr_name)
        setattr(self, '_'+attr_name, value)
        self.update_observers(self.__class__.__name__, attribute=attr_name,
                              old_value=oval, new_value=value)

if __name__ == '__main__':
    pass


