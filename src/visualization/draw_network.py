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
@title           :visualization/draw_network.py
@author          :ch
@contact         :christian@ini.ethz.ch
@created         :03/23/2017
@version         :1.0
@python_version  :3.5.2

Visualizing a NetworkModel instance.

Drawing the network structure of a NetworkModel instance. This can help to
verify the correctness of the implementation and to provide the user with a
better understanding of the implemented underlying network structures.
"""

import configuration as config

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

import logging
logger = logging.getLogger(config.logging_name)

def draw_network(network):
    """Draw the network structure that the user defined in the configuration
    file.

    Args:
        network: An instance of the class NetworkModel.

    Returns:
    """
    w = network.num_layers
    # Maximum number of neurons per layer.
    max_n = max([network.layer_size(i) for i in range(w)])

    # Width of a single layer.
    layer_width = 1./w
    # The maximum height a single neuron can demand.
    neuron_height = 1./max_n

    # Circle radius (radius of a single neuron).
    # Note, there must be twice as many neurons per layer (exc. plus inh.).
    radius = 0.8 * min(layer_width, .5*neuron_height) / 2.

    plt.clf()
    fig, ax = plt.subplots()

    # Store position of each neuron (index: layer, neuron, 'i' or 'e').
    positions = dict()

    ### Draw neurons of input layer.
    # The circle should be centered within a box of size layer_width x
    # neuron_height. Therefore, we need the offsets within this box.
    cox = layer_width / 2.
    coy = neuron_height / 2.

    nn = network.layer_size(0)

    x_offset = 0
    y_offset = (1 - nn * neuron_height) / 2.

    for n in range(nn):
        ind = (0, n, 'e')
        positions[ind] = (x_offset+cox, y_offset+coy)
        ax.add_patch(patches.Circle(positions[ind], radius, color='c'))

        y_offset += neuron_height

    ### Draw remaining neurons
    coyi = neuron_height / 4.
    coye = neuron_height * 3./4.

    for l in range(1, w):
        nn = network.layer_size(l)
        x_offset += layer_width
        y_offset = (1 - nn * neuron_height) / 2.

        for n in range(nn):
            indi = (l, n, 'i')
            inde = (l, n, 'e')
            positions[indi] = (x_offset+cox, y_offset+coyi)
            positions[inde] = (x_offset+cox, y_offset+coye)

            ax.add_patch(patches.Circle(positions[indi], radius, color='m'))
            ax.add_patch(patches.Circle(positions[inde], radius, color='c'))

            y_offset += neuron_height

    ### Draw connections of every layer.
    for l in range(1,w):
        _, _, eis, ies, ees = network.brian_objects(l)

        # For all Synapses from exc. neuron ni to inh. neuron nj.
        conn_ei = list(zip(eis.i, eis.j))
        if config.plot_network_partly:
            conn_ei = filter(lambda t: t[0] == 0 or t[0] == eis.N_pre-1,
                             conn_ei)
        for ni, nj in conn_ei:
            pi = positions[(l, ni, 'e')]
            pj = positions[(l, nj, 'i')]

            f = 1 if pi[1] < pj[1] else -1
            pi = (pi[0], pi[1]+f*radius)
            pj = (pj[0], pj[1]-f*radius)

            ax.add_patch(patches.FancyArrowPatch(pi, pj, color='0.6', \
                arrowstyle=patches.ArrowStyle.CurveA(head_length=radius*1.5,
                                                     head_width=radius)))



        # For all Synapses from inh. neuron ni to exc. neuron nj.
        conn_ie = list(zip(ies.i, ies.j))
        if config.plot_network_partly:
            conn_ie = filter(lambda t: t[0] == 0 or t[0] == ies.N_pre-1,
                             conn_ie)
        for ni, nj in conn_ie:
            pi = positions[(l, ni, 'i')]
            pj = positions[(l, nj, 'e')]

            f = 1 if pi[1] < pj[1] else -1
            pi = (pi[0], pi[1]+f*radius)
            pj = (pj[0], pj[1]-f*radius)

            ax.add_patch(patches.FancyArrowPatch(pi, pj, color='0.8', \
                connectionstyle=patches.ConnectionStyle.Arc3(rad=0.2), \
                arrowstyle=patches.ArrowStyle.CurveA(head_length=radius*1.5,
                                                     head_width=radius)))

        # For all Synapses from exc. neuron ni in layer l-1 to exc. neuron nj
        # in layer l.
        conn_ee = list(zip(ees.i, ees.j))
        if config.plot_network_partly:
            conn_ee = filter(lambda t: t[0] == 0 or t[0] == ees.N_pre-1,
                             conn_ee)
        for ni, nj in conn_ee:
            pi = positions[(l-1, ni, 'e')]
            pj = positions[(l, nj, 'e')]

            pi = (pi[0]+radius, pi[1])
            pj = (pj[0]-radius, pj[1])

            ax.add_patch(patches.FancyArrowPatch(pi, pj, \
                arrowstyle=patches.ArrowStyle.CurveA(head_length=radius*1.5,
                                                     head_width=radius)))

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # FIXME depends on the backend
    manager = plt.get_current_fig_manager()
    manager.resize(*manager.window.maxsize())


    if config.save_network_plot:
        # Check if directory of plot already exists.
        fn = config.network_plot_filename
        if not os.path.isdir(os.path.dirname(fn)):
            os.mkdir(os.path.dirname(fn))
        plt.savefig(fn, format='svg')



    if config.plot_network:
        plt.show()

if __name__ == '__main__':
    pass


