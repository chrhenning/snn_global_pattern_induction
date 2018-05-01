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
@title           :configuration.py
@author          :ch
@contact         :christian@ini.ethz.ch
@created         :03/15/2017
@version         :1.0
@python_version  :3.5.2

All the global configurations are specified here.

To costumize the simulation, this module provides the user a handy wrapper for
all configurations that can be made resp. the comments in this file guide to
external possibilities of costumization (e.g., Brian preferences or the
specification of the utilized neural dynamics).
"""

import logging
from datetime import datetime

########################################
### Dataset
########################################

# Which dataset shall be used?
# Must be one of the following options: ['mnist', '7segment']
dataset = 'mnist'

# Specifics for the datasets. Only change, if you modified the dataset
# (e.g.: changed the location). Hence, the remaining dataset options can be
# usually skipped.

### MNIST handwritten digit recognition.
# http://yann.lecun.com/exdb/mnist/

mnist_path = '../data/mnist/'

mnist_test_label   = mnist_path + 't10k-labels.idx1-ubyte'
mnist_test_images  = mnist_path + 't10k-images.idx3-ubyte'
mnist_train_label  = mnist_path + 'train-labels.idx1-ubyte'
mnist_train_images = mnist_path + 'train-images.idx3-ubyte'

# Dump read dataset to that file to allow faster reading later on.
mnist_pickle_dump  = mnist_path + 'mnist_dump.pickle'

### 7segment: LED Display Domain Data Set
# Recognition of digits on (noisy) 7-segment display.
# (Input is binary state for each of the 7 LEDs on the display.)
# http://archive.ics.uci.edu/ml/datasets/LED+Display+Domain
# Note, this is a toy dataset to test and evaluate the simulator with only a
# small amount of neurons.

seven_segment_path = '../data/7segment/'

seven_segment_data = seven_segment_path + '7segment.txt'

# The portion of samples that shall be used as training set (the rest will be
# used as test samples). Stratified sampling is utilized to balance the split.
seven_segment_split = .9

########################################
### Pattern Generation
########################################
# There are several options to generate patterns, that are induced to the
# network to strengthen distinct activations for different output classes.
# In addition, one can influence the way how the pattern is induced to the
# network.
# Note, that the pattern of the output layer will always be the identity
# matrix.
# Note, patterns aim to influence the activity of excitatory neurons.

# Choose the type of pattern, that defines the general form. The following
# options can be chosen:
# (Those options marked with an asterisk are purely positive, meaning there
# will be no external inhibitory influence on neurons when using these
# patterns.)
# -*'random': A random pattern would be drawn from a uniform distribution with
#   no further contraints.
# -*'sparse_random': A sparse random pattern would consist of uniformly drawn
#   sparse vectors, i.e., a random pattern with sparse activation.
# - 'dense_orthogonal': A dense and orthogonal pattern, would consist of
#   dense activation patterns in each layer, that would be orthogonal to one
#   another.
# -*'sparse_blurred': This pattern aims to approximate sparse orthogonal
#   activations within a layer. The algorithm starts with orthogonal canonical
#   unit vectors per output class. These unit vectors are blurred out in the
#   local neighborhood of the one entry according to a bell curve. Hence, a
#   blurry identity matrix with rank #num-output-classes is generated.
# - 'sparse_threshold': A matrix is generated according to the
#   'dense_orthogonal' pattern. Afterwards, all entries below a certain
#   threshold (or a certain percentage of entries) are set to zero.
# - 'max_mutual_equidistant': This pattern computes an optimal pattern
#   distribution, such that the mutual distance of patterns belonging to
#   different classes is maximized. Note, if the output layer has n neurons,
#   then no hidden layer may have less then n-1 neurons, otherwise the
#   condition cannot be fulfilled. The algorithm constructs an n dimensional
#   simplex, where each pair of vertices have the same distance. Subsequently,
#   the simplex is transformed such that all vertices lie on the surface of a
#   unit sphere. For details, refer to the documentation in the module
#   'src/util/simplex_construction'
#
# Note, all patterns implemented so far do not consider interlayer dependencies
# of patterns (which is probably unnecessary for fully-connected layers).
#
# All patterns are normalized, such that the class patterns in each layers are
# points on the unit sphere. Thus, patterns markes with an asterisk lay on the
# part of this surface that belongs to a single orthant.
#
# To understand what happens in the unlikely case, that a hidden layer has a
# dimension smaller than the number of output classes and orthogonality cannot
# be ensured (only for those methods that use orthogonalization), please refer
# to the documentation of the method extend_orthogonal_base in the module
# 'src/util/lin_alg'.
pattern_type = 'sparse_blurred'

# All patterns are generated, such that the pattern activation within a layer
# lies on a unit sphere. This fixed influence might be too strong or too weak,
# depending on the application. Therefore, one can scale the patterns using
# this option.
# Note, one can scale the pattern output-class-depending. Therefore, one should
# enter an array containing a scalar for each output class.
pattern_strength = 0.05


# In case the pattern is induced as external excitement to a postsynaptic
# neurons, where the synaptic weights are learned via standard STDP (e.g., when
# using the 'simple' induction method), the pattern induction might inhibit the
# induced pattern, as a post before pre spiking decreases the synaptic weight.
# In such a case we want to induce the inverted pattern (such that these are
# inhibited and the actual pattern is implicitly strengthened). Note, that
# lateral inhibition may then reduce the activity of favored neurons.
invert_patterns = False

## Settings specific to certain pattern types.

# The relative percentage (from 0 to 1) of neurons that should have a non-zero
# activation in their pattern in each layer.
# Can be a single value or array (layer-wise value for all hidden layers).
pattern_sparse_random_density = .1

# The standard deviation of the bell curve that is layed around the one entries
# of the identity matrix in the pattern "sparse_blurred".
# Can be a single value or array (layer-wise value for all hidden layers).
pattern_sparse_blurred_std_dev = 1

# The threshold for the "sparse_threshold" pattern. If a numerical value, its
# considered as the threshold, such that all values in the pattern below that
# value are set to zero. If its a numerical value inside a string, then it
# marks the percentage of values that should be set to zero (For instance:
# '0.1' would set the 10% of the smallest entries to zero in every activation
# vector for all layers and output classes).
pattern_sparse_threshold_thld = 0.01

########################################
### Pattern Visualization
########################################
# One can plot the activations of the generated patterns per hidden layer.
# There will be a plot for every output neuron! Additionally, some pattern
# statistics will be computed.
# Note, this visualization refers to raw patterns, there has been no scaling
# or inverting applied to them. Thus, the patterns do not illustrate there
# influence during simulation.

plot_patterns = False

# Additionally, one can save the figure.
save_pattern_plot = True
# The filename of the saved network plot.
patterns_plot_filename = '../plots/patterns.svg'

########################################
### Pattern Induction
########################################
# Specify how a pattern is induced into the network. Thus, how the generated
# pattern shall be used to stimulate the formation of neural assemblies
# corresponding to a certain input class.

# Whether or not the formation of neural assemblies shall be externally
# promoted using patterns.
# Note, if flag is deactivated, the simulation falls back to a standard
# unsupervised STDP learning, using the network as specified under the heading
# "Network Structure". I.e. patterns are not involved in the whole simulation
# process.
induce_patterns = False

# There are several possible ways on how to induce a pattern into the network.
# Currently implemented are the following.
# (In general, each layer is stimulated with an external pattern except for the
# input layer. Stimulation in layers is targeted to mainly affect the
# excitatory neurons in a layer. For each output class a specific pattern
# exists, which is only active if a sample of this output class is presented.)
# - 'simple': For each output class, there'll be a periodically firing neuron
#   connected to all excitatory neurons in the network as defined by the
#   pattern. The synaptic weight will be proportional to the value defined by
#   the pattern.
# - 'impact_modulation': STDP strengthens connections where a presynaptic spike
#   causes a postsynaptic one. To strengthen connections that may excite the
#   desired pattern, this method targets the way how synapses affect the
#   postsynaptic potential. Usually, on a presynaptic spike, a synapse modifies
#   the postsynaptic potential based on its weight. This method modulates this
#   modification by the pattern strength. See config
#   'impact_modulation_default' for details.
pattern_induction_method = 'impact_modulation'

# Finally, the network should work without external stimulation, as the
# external stimulation requires knowledge of class labels. Therefore, the
# influence should decay with an increasing number of epochs, such that the
# teacher signal finally vanishes.
# Here, one defines a value (<=1), by which the pattern is multiplied after
# each epoch.
pattern_induction_decay = 0.5

# To generalize well, specialization should only be reached in the higher
# layers (closer to the output). Lower layers should be free to adapt to the
# input distribution.
# Therefore, pattern induction in lower layers should be weaker as in higher
# layers.
# To achieve this, one can choose an adaption method:
# - 'linear': Linear decay of pattern influence (1*pattern for output layer,
#   0*pattern for input layer, linear interpolation in between).
# - 'quadratic': Quadratic decay of pattern influence.
#
# In addition, one might pass an arbitrary lambda expression, that gets as
# input the number of layers and the current layer. The output should be a
# single value that scales the pattern.
# Examples:
# - 'linear' = lambda l, L: l/(L-1)
# - 'quadratic = lambda l, L: (l/(L-1))**2
# Note, that the output layer has index L-1.
pattern_layer_influence = 'linear'

## Options specific for the induction methods.
# The firing rate of the pattern neurons of the induction method 'simple'.
# Unit of this option is considered as Hertz.
pattern_simple_fr = 63.75

## Settings specific to certain induction method.

# The method 'impact_modulation' is implemented due to an extra variable
# 'patmod' in the neural dynamics. This variable shall modulate the synaptic
# influence on a postsynaptic neuron according to the pattern strength. The
# default value of this variable is specified here and is applied if no pattern
# is influencing the network (i.e., if no training sample is presented). If the
# user has not specified this variable in the chosen neural dynamics, then the
# following equation will be added to the execution on a presynaptic spike at
# exc.-exc. synapses:
#   'v_post += patmod'
# and the default value will be forced to 0.
# Sensible (user-defined) update rules might either be multiplicative or
# additive. Examples:
#   'v_post += (w + patmod) * volt'          (default: 0)
#   'v_post += (w * patmod) * volt'          (default: 1)
#   'v_post += (w * (1 + patmod)) * volt'    (default: 0)
# Note, that the third multiplicative modulation should be preferred over the
# second one, as positive pattern values in the third one actually lead to a
# stronger excitation.
# Note, even if this induction method does not apply, but the variable 'patmod'
# appears in the neural dynamics, then the default value is applied to this
# variable the whole time.
impact_modulation_default = 0

########################################
### Network Structure
########################################
# To define the network structure, one basically determines the number of
# hidden layers and their shape. A hidden layer of size n has 2n neurons, as
# for every excitatory neuron one inhibitory neuron exists. The inhibitory
# neurons introduce lateral inhibition. Every excitatory neuron is connected to
# exactly one inhibitory neuron. Inhibitory to excitatory connections can be
# set up in two different ways. Either each inhibitory neuron is connected to
# all excitatory neurons of the layer except the incoming one or one specifies
# a window of size k, which means that only the k excitatory neurons to the
# left and the k ones to the right get connected. Note, that such local windows
# can emerge in the fully-connected case as well, since the weights still get
# learned.
# Note, that the output layer will have the same lateral inhibition structure
# if not specified otherwise.
# The connections in between layers will be fully-connected feedforward
# connections from excitatory to excitatory neurons.

# The number of hidden layers.
# Note, this parameter is only considered if the parameter 'hidden_layer_sizes'
# is a single value and not an array.
num_hidden_layers = 0

# The number of excitatory neurons per hidden layer.
# This parameter may either be a single value or an array of values. If one
# specifies a single number, then this will be the number of excitatory neurons
# in every hidden layer.
# If this parameter is an array, then each entry defines a new hidden layer and
# its number of excitatory neurons. Note, that this settings ignores the
# previously specified parameter 'num_hidden_layers'.
hidden_layer_sizes = 0 # 100

# The size of the output layer.
# Usually, the output layer has the size according to the number of classes.
# However, one may specify this differently. In such a case, the network output
# is classified due to an indirect measure. See 'Classification' options for
# details.
# If this option is set to None, then the Classification options are ignored
# and an output neuron is created for each class (the highest firing rate
# determines the class).
output_size = 400

# The size of the window, that marks the area of influence for an inhibitory
# neuron. There are 3 possible ways to define the behaviour. If the option is
# set to 'None', then an inhibitory neuron is connected to all excitatory
# neurons of that layer except the one that provides its input stimulus. If one
# specifies a single number k, then it connects to the 2k nearest neighbors.
# One may also specify an array of values, defining the window size in each
# layer (hidden + output layer).
# Note, if the window size if greater than half the number of neurons in a
# layer, then the inhibitory neuron is simply fully-connected to all excitatory
# neurons except for its own input neuron. If the window size is 0, then there
# will be no lateral inhibition (as well as no inhibitory neurons).
lateral_inhibition_window = None # [5, 3] # None

########################################
### Network Visualization
########################################
# One has the option to plot the network after it has been constructed. The
# simulation will freeze until the figure has been closed!
# Note, the plot won't contain feedback connections.
plot_network = False

# Drawing a fully-connected network may result in a large number of
# connections, that degrades readability and performance. Therefore, one may
# only want to draw the connections of some of the neurons in a homogenous
# network.
# If this option is activated, only the connections of the topmost and
# bottommost neuron in each layer are drawn.
plot_network_partly = False

# Additionally, one can save the figure.
save_network_plot = False
# The filename of the saved network plot.
network_plot_filename = '../plots/network.svg'

########################################
### Equations
########################################
# To specify the equations used to simulate neurons and synapses, one can
# specify a module name, that implements the neural dynamics. Therefore, one
# can choose a provided example model of neural dynamics or specify resp.
# costumize neural dynamics as a new module.
# Note, modules must reside in the 'equations' folder.
# The following example models are provided:
# - 'Diehl_Cook_SNN_Three_Factor': Implements conductance-based leaky-
#   integrate-and-fire neurons and three-factor STDP learning for exc.-exc.
#   synapses.
equation_module = 'Diehl_Cook_SNN_Three_Factor'
#equation_module = 'Simple_CurrLIF_and_STDP'
#equation_module = 'Kheradpisheh_NonLIF_STDP'

# If one wishes to costumize the equations, a copy of the file
# 'src/equations/equation_wrapper.py' has to be made and modified. Please refer
# to the documentation within this module to costumize the utilized neural
# dynamics.
# If a costumized module has been generated, it has to be imported by this
# module. Afterwards, the name of the module must be assigned to the variable
# 'equation_module'.

########################################
### Classification
########################################
# In case of a specified output layer size, one has to define here, how the
# network output is classified and evaluated.
# NOTE, if the output size of the network has not been specified, each class
# will be associated with a different output neuron, whose firing rate serves
# as classification criteria.

# Classification method. The classification method defines, how the output is
# assigned to a class label. The network might still be trained due to
# supervision (i.e., using pattern induction). This functionality allows more
# sophisticated network readouts. Note, that classifiers are trained on the
# most recent training batch. Hence, a training batch should not be too small.
# One of the following classification methods can be chosen:
#     - 'highest_response': Each output neuron is assigned to a class,
#       according to its highest response to class inputs during the
#       presentation of the last training batch.
#     - 'svm': A Support-Vector-Machine is trained on the firing rates. The SVC
#       implementation of Scikit-learn is utilized.
classification_method = 'highest_response'

## Configurations specific to a certain classification method.
# SVM penalty parameter of the error term. This can be single number or a
# list. In case of a list, grid search over that array will be applied. Note,
# that grid-search requires enough samples in the training batch to split the
# data into 3 folds.
# Example: svm_C = list(np.linspace(0.001,2, 1000))
svm_C = 1.0
# SVM kernel function. Must be one of 'linear', 'poly', 'rbf', 'sigmoid',
# 'precomputed' or a callable.
svm_kernel = 'rbf'

########################################
### Weight Initialization
########################################
# The method used to initialize synaptic weights.
# - 'uniform': Random matrix with uniformly distributed entries.
weight_initialization = 'uniform'

## Parameters specific for the chosen initialization method.
# Minimum und maximum weight for method 'uniform'.
winit_uniform_wmin = 0.
winit_uniform_wmax = .3

########################################
### Simulation Settings
########################################

### Input settings
# Minimum and maximum firing rate of an input neuron (the inputs will be scaled
# to that range).
# Note, input neurons have a Poisson distributed spiking train acc. to their
# firing rate.
input_fr_min = 0.0
input_fr_max = 63.75

# Maximum number of epochs. If training does not converge until this limit is
# reached, the simulation is stopped.
# If None is assigned, the simulation runs until convergence (or forever).
num_epochs = 1

# After how many epochs the validation error should be computed and reported.
# If no validation set is specified, the test set is used.
eval_interval = 2

# TODO: Better/ more general convergence criteria
# The validation error, at which convergence is reached. I.e. the relative
# percentage of samples that may be misclassified in the validation set.
validation_error = 0.01

# Resting and presentation period for a sample in ms.
# For each input presentation, there will be a short duration at the beginning
# to diminish dynamical effect from previous stimulations, i.e. a
# resting-period. Nothing is presented to the network to allow all neurons to
# reach their resting states. Afterwards, each sample is presented to the
# network for a certain amount of time.
resting_period = 150
presentation_period = 350

# Sometimes, no output spikes (and therefore no response) have been generated
# during the presentation period. Therefore, the presentation period can be
# repeated until sufficient output has been generated.
# The minimum number of output spikes, that have to be reached per sample
# presentation.
min_output_spikes = 5
# To strengthen the input for repetitions of the presentation period, one can
# define an increase of the input_fr_max, that is applied for every repetition,
# until the minimum number of output spikes have been reched.
input_fr_increment = input_fr_max / 2.
# To avoid that the network can ever be trapped in an infinite loop, a limit
# for the number of repetitions should be provided.
max_num_repetitions = 10

# Simulation timestep in ms.
simulation_timestep = 0.5

# Batch Size: Samples per Epoch.
# The training batch size determines how many samples are considered per Epoch.
# The validation Epoch determines how many samples are considered during
# evaluation. Final test results are always computed on the whole test set!
# If the batch size is specified as integer, it defines the absolute number of
# samples. If batch size is specified as float, it determines the relative
# percentage, i.e., the fraction of samples that shall be taken from the whole
# set (training or validation set).
# To use all samples per batch, use 1.
# The training batch is resampled every epoch, while the validation batch is
# hold fixed.
train_batch_size = 10 # 1.
val_batch_size = 50 # 1.

# After how many epochs a network snapshot (synaptic weight backup) should be
# generated. At the end, a final snapshot is generated.
# NOTE, snapshots from previous simulations will be deleted (i.e., the folder
# will be deleted)!
snapshot_every = 1
# The directory, where the snapshots shall be saved.
snapshot_dir = '../snapshots'

# How often to report how many samples have been processed so far.
# I.e., after how many presented samples a progress report should be logged.
feedback_interval = 20 # 100

########################################
### Brian Settings
########################################
# All Brian preferences can be set in the separate file called
# 'brian_preferences'. The settings made in this file will only affect this
# package.

########################################
### Recordings
########################################
# To understand the effects of the chosen equations and hyperparameters, it is
# often useful to carefully consider the development of state variables over
# time at certain neurons/synapses.
# The recorded values are stored into a JSON file.

# Whether or not to plot the recordings while simulating.
# FIXME not yet implemented
online_recording = False

# Whether to save plots of the recordings or not.
save_recording_plots = True

# Where to store the recorded values and their plots?
# There will be a subdirectory for each recording.
# NOTE, previous recordings and their plots will be deleted (i.e., the folder
# will be deleted)!
recording_dir = '../plots/recordings'

### State variable recordings (StateMonitors).
# Define, which synaptic or neuronal state variables shall be recorded during
# simulation.
# Each recording is a tuple of the following form:
#   (type (str), layer (int), variables, indices, dt, plot_duration)
# Individual fields have the following meaning:
# - type: The type of neuron or synapses, i.e.:
#   - 'ne': Excitatory neurons
#   - 'ni': Inhibitory neurons
#   - 'ee': Exc.-exc. synapses
#   - 'ei': Exc.-inh. synapses
#   - 'ie': Inh.-exc. synapses
# - layer: The index of the layer, where to record from (0 is input layer). Use
#   layer i+1 for exc.-exc. connections from layer i to layer i+1.
# - variables: A string or a list of strings, each being the name of a state
#   variable.
# - indices: An int or a list of ints, each determining the index of the entity
#   (neuron or synapse) to record from.
# - dt: The timestep in milliseconds at which recordings are taken. If None,
#   the simulation timestep is used.
# - plot_duration: The timespan per plot in milliseconds. I.e. the whole
#   simulation time is divided into sections of size plot_duration and a plot
#   is generated for each section.
# Example: ('ne', 1, 'v', [0], None, 1000) - Record voltage from first
# excitatory neuron in layer 1 (first hidden layer). Record a value every
# simulations timestep. Create a new plot every second.
state_var_recordings = [
    ('ne', 0, ['rates'], [0], None, 1000),
#    ('ne', 1, ['v'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,13,14], None, 1000),
    ('ne', 1, ['v'], [0,1,2,3,4,5,6,7,8,9], None, 1000),
    ('ni', 1, ['v'], [0,1,2,3,4,5,6,7,8,9], None, 1000),
#    ('ni', 2, ['v'], [0, 1, 2], None, 1000),
    ('ee', 1, ['w'], [0, 1, 15, 30], None, 1000),
#    ('ee', 2, ['w'], [0, 1, 15, 30], None, 1000)
]

### Population Rate Recordings (PopulationRateMonitor).
# Measure the instantaneous firing rate of a NeuronGroup averaged accross all
# neurons in that group. I.e. for each simulated timestep the firing rate is
# recomputed for all spikes that have occurred in that timestep.
# Each recording is a tuple of the following form:
#   (type (str), layer (int), plot_duration, smooth_window, smooth_width)
# - type: The type of neuron:
#   - 'ne': Excitatory neurons
#   - 'ni': Inhibitory neurons
# - layer: The index of the layer, where to record from (0 is input layer).
# - plot_duration: The timespan per plot in milliseconds. I.e. the whole
#   simulation time is divided into sections of size plot_duration and a plot
#   is generated for each section.
# - For plots (not stored recordings) the smooth_rate method of Brian class
#   PopulationRateMonitor can be applied. If smooth_window is None, no
#   smoothing will be applied. smooth_width is given in milliseconds.
# Example: ('ne', 1, 1000, None, None) - Record the instantaneous firing rates
# from excitatory neurons in layer one. Do not smooth the firing rates when
# plotting.
population_rate_recordings = [
    ('ne', 0, 1000, 'gaussian', 1),
    ('ne', 1, 1000, 'gaussian', 1),
#    ('ne', 2, 1000, 'gaussian', 1)
]

### Record Spike Events (SpikeMonitor).
# Record spike events and optionally state variables at spike events during
# simulation. This will always record for all neurons in the NeuronGroup.
# Each recording is a tuple of the following form:
#   (type (str), layer (int), variables, plot_duration)
# - type: The type of neuron:
#   - 'ne': Excitatory neurons
#   - 'ni': Inhibitory neurons
# - layer: The index of the layer, where to record from (0 is input layer).
# - variables: A string or a list of strings, each being the name of a state
#   variable. At each spike event, their value is recorded. If only spike times
#   should be recorded, use None.
# - plot_duration: The timespan per plot in milliseconds. I.e. the whole
#   simulation time is divided into sections of size plot_duration and a plot
#   is generated for each section.
# Example: ('ne', 1, None, 1000) - Record the spike events for each excitatory
# neuron in layer one. Only spike time for neurons is recorded no other state
# variables.
spike_event_recordings = [
    ('ne', 0, 'rates', 1000),
    ('ne', 1, 'v', 1000),
 #   ('ne', 2, 'v', 1000),
    ('ni', 1, 'v', 1000),
 #   ('ni', 2, 'v', 1000)
]

########################################
### Evaluation
########################################
# Directory, where to store simulation results.
# NOTE, previous evaluations will be deleted (i.e., the folder will be
# deleted)!
eval_dir = '../evaluation'

# Whether or not to store the results of the output evaluation (i.e.,
# accuracies, f-scores) during training. This would store the training scores
# at each epoch and the validation scores (test scores, if no validation set
# available) as specified by parameter 'eval_interval'.
# False would lead to storing only final test scores.
store_output_eval_progress = True

########################################
### Other Settings
########################################

# Set a random seed for all random processes during the simulation to allow
# reproducibility of results.
random_seed = 42

# You may utilize multithreading to speed up parts of the simulation.
# You can either set a maximum number of threads to use (1 would mean no
# multithreading) or you can let the simulator choose the optimal maximum
# number (number of virtual cpu cores).
# Note, parallelization is only utilized in some parts of the simulation
# pipeline (e.g., network generation).
# Note, this does not enable multithreading in Brian. See Brian settings for
# details.
# Note, the current implementation uses multithreading not multiprocessing. As
# a python interpreter can only execute one thread at a time, this option might
# not lead to significant speedup (only underlying C implementations are
# parallelized).
num_threads = None

# Set log level (Choose one of the following: DEBUG, INFO, WARNING, ERROR,
# CRITICAL).
file_loglevel = logging.DEBUG
console_loglevel = logging.DEBUG

# Determing the name of the log file (all console outputs will be written to
# that file). If you assign an empty string, no logfile will be generated.
logging_logfile = '../logs/' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + \
                  '_snn_spd.log'

# The name of the logger that is configured. (There is no reason to change this
# config.)
logging_name = 'snn_stp_logger'

# Profile the simulation.
simulation_profiling = False

if __name__ == '__main__':
    pass

