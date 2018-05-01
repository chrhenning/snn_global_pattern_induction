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
@title           :recordings.py
@author          :ch
@contact         :christian@ini.ethz.ch
@created         :04/24/2017
@version         :1.0
@python_version  :3.5.2

This class takes care of recording state variables during simulation.
"""

import configuration as config
from util.config_exception import ConfigException
from util import utils
from pypatterns.singleton import Singleton
from pypatterns.observer import Observer
from simulation import Simulation

import brian2 as b2
import os
import _pickle as pickle
import matplotlib.pyplot as plt
import numpy as np

import logging
logger = logging.getLogger(config.logging_name)

class Recordings(Observer, metaclass=Singleton):
    """To understand the dynamics of a network, its internal state and
    variables must be recordable. This class shall allow one to analyse the
    dynamics during and after simulation according to the configs.

    Under the hood, this class simply creates instances of the Brian classes
    StateMonitor, PopulationRateMonitor and SpikeMonitor. However,
    SpikeMonitors are not instantiated in this class due to efficiency
    considerations. Moreover, the instantiation in the class NetworkModel is
    modified. This introduces an unneccesary interweaving of recording code
    with simulation code, but ensures that SpikeMonitors are only instantiated
    once.

    Attributes:
    """
    def __init__(self, network):
        """Generate all recording objects and add them to the network.

        Note, SpikeMonitors have been already instantiated.

        Args:
            network: An instance of class NetworkModel.

        Returns:
        """
        super().__init__()

        self._network = network

        Recordings._check_state_var_recordings()
        Recordings._check_population_rate_recordings()
        Recordings._check_spike_event_recordings()

        # FIXME Following two methods are dirty and misplaced.
        # Check that chosen layers exist (cannot be done in static methods, as
        # network has to be known).
        def layer_exists(tup, layer):
            if not (layer >=0 and layer < network.num_layers):
                raise ConfigException('Recording %s has non-existing layer.' \
                                      % (str(tup)))

        # Make sure, indices and vars exist in layer.
        def vars_exists(tup, layer, source, var, inds=None):
            if var is not None and isinstance(var, list):
                for v in var:
                    if not hasattr(source, v):
                        print(source.get_states().keys())
                        raise ConfigException('Variable ' + v + ' does not ' \
                                              + 'exist from recording %s.' \
                                              % (str(tup)))
            if inds is not None and isinstance(inds, list):
                for i in inds:
                    if not (i >=0 and i < source.N):
                        raise ConfigException('Recording %s cannot have ' \
                                      % (str(tup)) + 'index %d.' % (i))

        self._state_monitors = dict()
        self._pop_rate_monitors = dict()
        self._spike_monitors = dict()

        for tup in config.state_var_recordings:
            typ, layer, var, inds, dt, _ = tup
            layer_exists(tup, layer)

            exn, inn, eis, ies, ees = network.brian_objects(layer)

            source = None

            if typ == 'ne':
                source = exn
            elif typ == 'ni':
                source = inn
            elif typ == 'ei':
                source = eis
            elif typ == 'ie':
                source = ies
            else:
                source = ees
            vars_exists(tup, layer, source, var, inds)

            dt = dt * b2.ms if dt is not None else dt
            state_mon = b2.StateMonitor(source, var, inds, dt=dt)
            self._state_monitors[str(tup)] = state_mon
            network.add_component(state_mon)

        for tup in config.population_rate_recordings:
            typ, layer, _, _, _ = tup
            layer_exists(tup, layer)
            exn, inn, _, _, _ = network.brian_objects(layer)

            source = None

            if typ == 'ne':
                source = exn
            else:
                source = inn

            pop_rmon = b2.PopulationRateMonitor(source)
            self._pop_rate_monitors[str(tup)] = pop_rmon
            network.add_component(pop_rmon)

        for tup in config.spike_event_recordings:
            typ, layer, var, _ = tup
            layer_exists(tup, layer)

            sp_mon = None

            if typ == 'ne':
                sp_mon = network.exc_spike_monitor(layer)
            else:
                sp_mon = network.inh_spike_monitor(layer)
            vars_exists(tup, layer, sp_mon.source, var)

            self._spike_monitors[str(tup)] = sp_mon

        # For online recordings, we need to know, when the network state has
        # changed.
        if config.online_recording:
            sim = Simulation()
            sim.register(self)

    def update(self, *args, **kwargs):
        """Update plots for online recordings.

        TODO: In future, one could incrementally write recordings to a file.

        Args:

        Returns:
        """
        if args[0] == 'Simulation':
            # TODO online plotting of recordings.
            #print(kwargs['curr_sim_time'])
            pass
        else:
            assert(False)

    """
    Static class attribute, that contains the attributes passed to
    SpikeMonitors.
    """
    _spike_monitor_args = None

    def store_recordings(self):
        """Store the whole recordings made during simulation into files and
        optionally into plots.

        Args:

        Returns:
        """
        plt.close('all')

        if not os.path.isdir(config.recording_dir):
            os.makedirs(config.recording_dir)

        ### Handle StateMonitors.
        for tup in config.state_var_recordings:
            state_mon = self._state_monitors[str(tup)]

            typ, layer, var, inds, dt, duration = tup
            var_str = utils.list_to_str(var)
            inds_str = '_'+str(inds) if isinstance(inds, bool) \
                else utils.list_to_str(inds)

            folder_name = 'state_monitor_%s_%d_vars%s_indices%s_%s_%d' \
                          % (typ, layer, var_str, inds_str, str(dt), duration)
            folder_name = os.path.join(config.recording_dir, folder_name)

            os.mkdir(folder_name)
            logger.info("StateMonitor recordings %s are stored in %s." \
                        % (str(tup), folder_name))

            dump_obj = dict()
            dump_obj['type'] = typ
            dump_obj['layer'] = layer
            dump_obj['variables'] = var
            dump_obj['indices'] = inds
            dump_obj['dt'] = dt
            dump_obj['recordings'] = dict()

            recs = dump_obj['recordings']
            recs['t'] = np.array(getattr(state_mon, 't_'))

            for v in var:
                recs[v] = np.array(getattr(state_mon, '%s_' % v))
                assert(len(recs[v].shape) == 2)

            # Store recordings in file.
            dump_file = os.path.join(folder_name, 'recordings.pickle')
            with open(dump_file, 'wb') as f:
                pickle.dump(dump_obj, f)

            # Generate recording plots.
            if config.save_recording_plots:
                # Note, that duration is in ms, but as recs['t'] is
                # dimensionless, its values are interpretable as seconds.
                slice_gen = utils.list_to_val_dependent_slices(recs['t'],
                                                               duration/1000)
                # For each slice, a variables with all its recorded indices
                # will be part of a plot (one plot for each duration and
                # variable).

                # Compute min and max to scale y-axis uniformly per var.
                mins = dict()
                maxs = dict()
                for v in var:
                    mins[v] = np.min(recs[v])
                    maxs[v] = np.max(recs[v])

                for sind, eind in slice_gen:
                    for v in var:
                        # Note, that inds might be boolean.
                        ind_labels = inds
                        if not isinstance(inds, list):
                            ind_labels = list(range(recs[v].shape[0]))
                        vunit = getattr(state_mon.source,v).unit

                        Recordings._plot_slice(recs['t'], recs[v], sind, eind,
                                               v, ind_labels, str(tup), vunit,
                                               folder=folder_name,
                                               miny=mins[v], maxy=maxs[v])

        ### Handle PopulationRateMonitors.
        for tup in config.population_rate_recordings:
            prate_mon = self._pop_rate_monitors[str(tup)]

            typ, layer, duration, swin, swidth = tup

            folder_name = 'pop_rate_monitor_%s_%d_%d_%s_%s' \
                          % (typ, layer, duration, str(swin), str(swidth))
            folder_name = os.path.join(config.recording_dir, folder_name)
            os.mkdir(folder_name)
            logger.info("PopulationRate recordings %s are stored in %s." \
                        % (str(tup), folder_name))

            dump_obj = dict()
            dump_obj['type'] = typ
            dump_obj['layer'] = layer
            dump_obj['t'] = np.array(getattr(prate_mon, 't_'))
            dump_obj['rate'] = np.array(getattr(prate_mon, 'rate_'))

            # Store recordings in file.
            dump_file = os.path.join(folder_name, 'recordings.pickle')
            with open(dump_file, 'wb') as f:
                pickle.dump(dump_obj, f)

            # Generate recording plots.
            if config.save_recording_plots:
                slice_gen = utils.list_to_val_dependent_slices(dump_obj['t'],
                                                               duration/1000)

                if swin is not None:
                    rates = np.array(prate_mon.smooth_rate(swin, swidth*b2.ms))
                else:
                    rates = dump_obj['rate']

                # Compute min and max to scale y-axis uniformly for rates.
                miny = np.min(rates)
                maxy = np.max(rates)

                for sind, eind in slice_gen:
                    Recordings._plot_slice(dump_obj['t'], rates, sind, eind,
                                           'rate', None, str(tup), b2.Hz,
                                           folder=folder_name, miny=miny,
                                           maxy=maxy)

        ### Handle SpikeMonitors.
        for tup in config.spike_event_recordings:
            spike_mon = self._spike_monitors[str(tup)]

            typ, layer, var, duration = tup
            var_str = '_None' if var is None else utils.list_to_str(var)

            folder_name = 'spike_monitor_%s_%d_vars%s_%d' \
                          % (typ, layer, var_str, duration)
            folder_name = os.path.join(config.recording_dir, folder_name)
            os.mkdir(folder_name)
            logger.info("Spike recordings %s are stored in %s." \
                        % (str(tup), folder_name))

            dump_obj = dict()
            dump_obj['type'] = typ
            dump_obj['layer'] = layer
            dump_obj['recordings'] = dict()

            recs = dump_obj['recordings']

            recs['t'] = np.array(getattr(spike_mon, 't_'))
            recs['i'] = np.array(getattr(spike_mon, 'i_'))

            if var is not None:
                for v in var:
                    recs[v] = np.array(getattr(spike_mon, '%s_' % v))

            # Store recordings in file.
            dump_file = os.path.join(folder_name, 'recordings.pickle')
            with open(dump_file, 'wb') as f:
                pickle.dump(dump_obj, f)

            # Generate recording plots.
            if config.save_recording_plots and len(recs['i']) > 0:
                # We need to keep track of the time to scale the x-axis.
                # etime = stime + duration/1000
                stime = 0
                slice_gen = utils.list_to_val_dependent_slices(recs['t'],
                                                               duration/1000)

                # Compute min and max values to properly and uniformly color
                # code vars.
                if var is not None:
                    mins = dict()
                    maxs = dict()
                    for v in var:
                        mins[v] = np.min(recs[v])
                        maxs[v] = np.max(recs[v])

                # We need to know the number of neurons, to set ymax.
                ymin = -0.5
                ymax = spike_mon.source.N - 0.5

                for sind, eind in slice_gen:
                    minx = stime
                    maxx = minx + duration/1000
                    stime = maxx

                    # Plot pure spike events.
                    Recordings._scatter_slice(recs['t'], recs['i'], sind, eind,
                                              minx, maxx, ymin, ymax, str(tup),
                                              folder=folder_name)

                    if var is None:
                        continue

                    for v in var:
                        vunit = getattr(spike_mon,v).unit
                        Recordings._scatter_slice(recs['t'], recs['i'], sind,
                                                  eind, minx, maxx, ymin, ymax,
                                                  str(tup), var=recs[v],
                                                  var_min=mins[v],
                                                  var_max=maxs[v], var_name=v,
                                                  var_unit=vunit,
                                                  folder=folder_name)
            elif len(recs['i']) == 0:
                logger.warning('Could not generate Plots for SpikeMonitor ' \
                               + 'recordings %s. No spike events.' \
                               % (str(tup)))

    @staticmethod
    def get_spike_monitor_args(layer):
        """As SpikeMonitors are not instantiated in this class (as they are
        needed by the network anyway, in order to compute firing rates), we
        need to let the network know, which arguments it has to pass to the
        SpikeMonitors it generates.

        Args:
            layer: The method returns the SpikeMonitor arguments for the
                excitatory and inhibitory neurons in a specific layer.

        Returns:
            A tuple of tuples (actually lists). The returned list has the
            following shape
                [[exc_variables, exc_record], [inh_variables, inh_record]]
        """
        if Recordings._spike_monitor_args is None:
            Recordings._check_spike_event_recordings()

            Recordings._spike_monitor_args = dict()
            for tup in config.spike_event_recordings:
                t, l, var, duration = tup

                Recordings._spike_monitor_args.setdefault(l, [[None, False],
                                                              [None, False]])

                te, ti = Recordings._spike_monitor_args[l]
                curr_tup = None
                if t == 'ne':
                    curr_tup = te
                else:
                    curr_tup = ti

                curr_tup[1] = True

                if var is not None:
                    if curr_tup[0] is None:
                        curr_tup[0] = []
                    curr_tup[0].extend(var)

        Recordings._spike_monitor_args.setdefault(layer, [[None, False],
                                                          [None, False]])
        return Recordings._spike_monitor_args[layer]

    @staticmethod
    def _plot_slice(time, var, sind, eind, var_name, ind_names, title, unit,
                    folder=None, miny=None, maxy=None):
        """Plot a time slice for a variable recording.

        Args:
            time: The time array.
            var: The recorded values.
            sind: Start index of slice.
            eind: End index of slice.
            var_name: Name of the recorded variable.
            ind_names: Names of the recorded indices.
            title: Plot titel.
            unit: Variable unit.
            folder: Where to store plot.
            miny: Minimum y limit.
            maxy: Maximum y limit.

        Returns:
        """
        if len(var.shape) == 2:
            for i in range(var.shape[0]):
                label = '%s_%d' % (var_name, ind_names[i])

                values = var[i,sind:eind]
                b2.plot(time[sind:eind], values, label=label)
        else:
            label = var_name
            values = var[sind:eind]
            b2.plot(time[sind:eind], values, label=label)

        plt.legend()
        plt.title(title)
        plt.xlabel('time (seconds)')
        unit = str(unit)
        if unit == 'rad':
            plt.ylabel('%s' % (var_name))
        else:
            plt.ylabel('%s (%s)'  % (var_name, unit))
        axes = plt.gca()
        # Make sure one also can see min and max vals in plot.
        if miny is not None and maxy is not None:
            eps = 0.01 * (maxy - miny)
            miny -= eps
            maxy += eps
        if miny is not None:
            axes.set_ylim(bottom=miny)
        if maxy is not None:
            axes.set_ylim(top=maxy)
        if folder is not None:
            plot_name = 'plot_%s_%d_%d.png' % (var_name, sind, eind)
            plot_name = os.path.join(folder, plot_name)
            plt.savefig(plot_name)
        plt.close()


    @staticmethod
    def _scatter_slice(time, neurons, sind, eind, xmin, xmax, ymin, ymax,
                       title, y_label='Neuron i', var=None, var_min=None,
                       var_max=None, var_name=None, var_unit='rad',
                       folder=None):
        """Create a scatter plot of a time slice for a spike recording.
        Variables recorded on spike events might be plottet as color coded
        points.

        Args:
            time: The time array.
            neurons: Which neuron spiked at a time point.
            sind: Start index of slice.
            eind: End index of slice.
            xmin: Start Time (needed to set x-range).
            xmax: End Time (needed to set x-range).
            ymin: Lowest y-value.
            ymax: Highest y-value.
            title: Plot titel.
            ylabel: y label.
            var: An optional color coded variable, that was recorded on spike
                events.
            var_min: Minimal var value.
            var_max: Maximum var value.
            var_name: Name of var.
            var_unit: Unit of var.
            folder: Where to store plot.

        Returns:
        """
        if var is None:
            b2.scatter(time[sind:eind], neurons[sind:eind])
        else:
            cm = plt.cm.get_cmap('coolwarm')
            sp = b2.scatter(time[sind:eind], neurons[sind:eind],
                            c=var[sind:eind], vmin=var_min, vmax=var_max,
                            cmap=cm)

            cb = plt.colorbar(sp)
            if var_name is not None:
                unit = str(var_unit)
                if unit == 'rad':
                    cb.ax.set_xlabel('%s' % (var_name))
                else:
                    cb.ax.set_xlabel('%s (%s)'  % (var_name, unit))

        axes = plt.gca()
        axes.set_xlim(left=xmin, right=xmax)
        axes.set_ylim(bottom=ymin, top=ymax)
        plt.title(title)
        plt.xlabel('time (seconds)')
        plt.ylabel(y_label)
        if folder is not None:
            if var_name is None:
                plot_name = 'plot_spikes_%d_%d.png' % (sind, eind)
            else:
                plot_name = 'plot_spikes_%s_%d_%d.png' % (var_name, sind, eind)
            plot_name = os.path.join(folder, plot_name)
            plt.savefig(plot_name)
        plt.close()

    @staticmethod
    def _check_state_var_recordings():
        """Assert that the option state_var_recordings is properly defined.

        Args:

        Returns:
        """
        for i, tup in enumerate(config.state_var_recordings):
            if not isinstance(tup, tuple) or len(tup) != 6:
                err_msg = 'Option \'state_var_recordings\' should be ' \
                              + ' a list of tuples of size 6.'
                raise ConfigException(err_msg)

            t, l, var, inds, _, _ = tup
            types = ['ne', 'ni', 'ee', 'ei', 'ie']
            if not (t in types and isinstance(l, int) and l >= 0 \
                    and (isinstance(var, str) or isinstance(var, list)) \
                    and (isinstance(inds, (bool, int)) \
                         or isinstance(inds, list))):
                err_msg = 'The tuple %s from option ' % (str(tup))\
                          + '\'state_var_recordings\' is not properly ' \
                          + 'formated.'
                raise ConfigException(err_msg)

            if isinstance(var, str):
                config.state_var_recordings[i] = \
                    utils.set_tuple_item(tup, 2, [var])
            if isinstance(inds, int) and not isinstance(inds, bool):
                config.state_var_recordings[i] = \
                    utils.set_tuple_item(tup, 3, [inds])

    @staticmethod
    def _check_population_rate_recordings():
        """Assert that the option population_rate_recordings is properly
        defined.

        Args:

        Returns:
        """
        for tup in config.population_rate_recordings:
            if not isinstance(tup, tuple) or len(tup) != 5:
                err_msg = 'Option \'population_rate_recordings\' should be ' \
                              + ' a list of tuples of size 5.'
                raise ConfigException(err_msg)

            t, l, _, _, _ = tup
            if not (t in ['ne', 'ni'] and isinstance(l, int) and l >= 0):
                err_msg = 'The tuple %s from option ' % (str(tup))\
                          + '\'population_rate_recordings\' is not properly ' \
                          + 'formated.'
                raise ConfigException(err_msg)

    @staticmethod
    def _check_spike_event_recordings():
        """Assert that the option spike_event_recordings is properly defined.

        Args:

        Returns:
        """
        for i, tup in enumerate(config.spike_event_recordings):
            if not isinstance(tup, tuple) or len(tup) != 4:
                err_msg = 'Option \'spike_event_recordings\' should be ' \
                              + ' a list of tuples of size 4.'
                raise ConfigException(err_msg)

            t, l, var, _ = tup
            if not (t in ['ne', 'ni'] and isinstance(l, int) and l >= 0 \
                    and (var is None or isinstance(var, str)
                         or isinstance(var, list))):
                err_msg = 'The tuple %s from option ' % (str(tup))\
                          + '\'spike_event_recordings\' is not properly ' \
                          + 'formated.'
                raise ConfigException(err_msg)

            if isinstance(var, str):
                config.spike_event_recordings[i] = \
                    utils.set_tuple_item(tup, 2, [var])

if __name__ == '__main__':
    pass


