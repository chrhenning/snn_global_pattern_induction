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
@title           :evaluation.py
@author          :ch
@contact         :christian@ini.ethz.ch
@created         :04/19/2017
@version         :1.0
@python_version  :3.5.2

The module provides several methods to evaluate the simulation results.

Please refer to the description of the static methods defined in this module
"""

import configuration as config
from util.config_exception import ConfigException
from readout.svm import SVM
from readout.highest_response import HighestResponse

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import numpy as np
from collections import Counter
import os
import json
import time

import logging
logger = logging.getLogger(config.logging_name)

"""Contains the last trained readout classifier. Every time a new training
batch is evaluated, this classifier is retrained. (Note, this only applies for
alternative classifiers. The standard readout demands no additional training.
"""
CLASSIFIER = None

def evaluate_output_rates(rates, labels, num_classes, tag, store=True,
                          epoch=None):
    """Evaluate the firing rates of the output layer.

    If the output layer size has not been specified by the user, each output
    neuron corresponds to a different class and the maximum firing rate
    determines the predicted label. If the user wishes to use another readout
    mechanism on the customized output layer, this method is used instead. An
    overview of alternative readout methods is given in the config file.

    This method computes measures for multiclass classification problems, such
    as accuracy and f-scores. The scores are logged. If desired, the scores can
    also be stored in a file (with more details). The filename will be:
        'output_eval_<tag>.jsonl'

    Args:
        rates: A list of arrays. Each entry is an 1D array of spike counts
            (from the output layer).
        labels: A list of ground truth labels. Each entry corresponds to the
            rates given in the first parameter.
        num_classes: The number of output classes.
        tag: A name, that is associated with the data split, that is evaluated.
            This tag will define the filename. If an alternative readout
            classifier is used, this tag decides, wether this classifier has
            to be retrained.
        store: Whether to store the results or not. Results are appended as a
            single JSON object to a JSONL file.
        epoch: If specified, the epoch can later help to identify the results.

    Returns:
        The error (1-accuracy). (Or 1, if nothing can be evaluated)
    """

    y_true = [] # Gound truth
    y_pred = [] # Predicted sample

    rates_arr = np.asarray(rates)
    labels_arr = np.asarray(labels)

    # We want to ignore those samples, that elicited no output spiking. (This
    # is decided by the user, as he can tweak the simulation to continue as
    # long as no output spikes appeared).
    # I.e., if all output rates are zero, the sample is ignored.
    valid_rates = np.where(np.sum(rates_arr, axis=1) != 0)[0]
    ignored = labels_arr.size - valid_rates.size
    # Delete invalid samples.
    rates_arr = rates_arr[valid_rates]
    labels_arr = labels_arr[valid_rates]
    # Simple renaming.
    y_true = labels_arr

    # Helper functions to print results.
    _perc = lambda x : round(100*x,2)
    _sperc = lambda x : '%.2f%%' % (_perc(x))

    if ignored > 0:
        logger.warning('%d (%s) samples have been ignored during ' \
                       % (ignored, _sperc(ignored/len(rates))) \
                       + 'evaluation, since there has been no output ' \
                       + 'activity.')

    # Just a flag that is set to False for alternative classifiers, as they
    # provide less evaluation metrics. (I.e., some metrics cannot be calculated
    # and stored when using alternative classifiers).
    extra_scores = False

    # If we need no special readout. Meaning each output neuron simply
    # represents a class and no further logic is needed to classify the output.
    # Note, that this option provides several eval measures (such as
    # ambiguousness) that the alternative classifiers do not provide.
    if config.output_size is None:
        extra_scores = True

        # Predicted samples, where ambiguous outcomes are not necessarily
        # misclassification. I.e., if the correct class has one of the highest
        # outputs, it is considered as correct classification.
        y_pred_with_ambig = []
        confidence = [] # Confidence of correct output
        unambiguousness = [] # Normalized distance to second best prediction
        # Ambiguous samples are samples, that have multiple outputs with the
        # same maximum firing rate. They are considered as misclassifications.
        ambiguous = 0

        for i in range(rates_arr.shape[0]):
            frates = rates_arr[i,:]
            label = labels_arr[i]

            pred = np.argmax(frates)
            pred_with_ambig = pred
            # Ensure, that output is not ambiguous.
            counts = Counter(frates)
            if counts[frates[pred]] > 1:
                ambiguous += 1
                # Choose index, that is not the correct one, but has maximum
                # confidence. I.e., enforce misclassification.
                pred = np.argmax(np.concatenate((frates[:label], [-1],
                                                 frates[label+1:])))
                assert(pred != label)

                if frates[pred] == frates[label]:
                    pred_with_ambig = label

            norm_frates = frates / np.linalg.norm(frates, 1)
            sec_best = np.argmax(np.concatenate((frates[:pred],
                                                 frates[pred+1:])))

            y_pred.append(pred)
            y_pred_with_ambig.append(pred_with_ambig)

            confidence.append(norm_frates[label])
            unambiguousness.append(norm_frates[pred] - norm_frates[sec_best])

        y_pred = np.asarray(y_pred)

    # Use an alternative readout classifier to evaluate the output rates.
    # Retrain the classifier if tag == 'training'.
    else:
        global CLASSIFIER

        # Retrain if necessary.
        if tag == 'training':
            logger.debug('Retraining readout classifier according to method:' \
                         + ' \'%s\'' % config.classification_method)
            if config.classification_method == 'highest_response':
                CLASSIFIER = HighestResponse()
                CLASSIFIER.fit(rates_arr, labels_arr, num_classes=num_classes)
            elif config.classification_method == 'svm':
                CLASSIFIER = SVM()
                CLASSIFIER.fit(rates_arr, labels_arr, C=config.svm_C,
                               kernel=config.svm_kernel)
            else:
                error_msg = 'Classification method \'%s\' is unknown. ' \
                        % config.classification_method
                raise ConfigException(error_msg)

        # Predict outcome for given rates.
        y_pred = CLASSIFIER.predict(rates_arr)

    if y_true.size == 0:
        return 1

    json_obj = dict()
    json_obj['timestamp'] = time.time()
    if epoch is not None:
        json_obj['epoch'] = epoch
    json_obj['num_samples'] = len(rates)
    json_obj['ignored'] = ignored
    json_obj['classification_method'] = None
    if config.output_size is not None:
        json_obj['classification_method'] = config.classification_method

    if extra_scores:
        json_obj['ambiguous'] = ambiguous

        if ambiguous > 0:
            logger.debug('%d (%s) samples had more than one output neuron with'\
                         % (ambiguous, _sperc(ambiguous/len(rates)))
                         + ' maximum confidence (ambiguous classification).')

    acc = accuracy_score(y_true, y_pred)
    json_obj['accuracy'] = acc
    logger.info('### %s accuracy: %s' % (tag, _sperc(acc)))

    if extra_scores and ambiguous > 0:
        acc_with_ambig = accuracy_score(y_true, y_pred_with_ambig)
        if acc_with_ambig != acc:
            json_obj['accuracy_with_ambiguous'] = acc_with_ambig
            logger.info('When ambiguous outcomes are allowed, the accuracy ' \
                        + 'would be: %s' % (_sperc(acc_with_ambig)))

    classes = list(range(num_classes))

    def _f_score(method):
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, \
            labels=classes, average=method)
        json_obj['prec_'+method] = prec
        json_obj['rec_'+method] = rec
        json_obj['f1_'+method] = f1
        return f1

    f1_micro = _f_score('micro')
    f1_macro = _f_score('macro')
    f1_weighted = _f_score('weighted')

    logger.info('Micro/Macro/Weighted - F-Scores: %.4f, %.4f, %.4f.' \
                % (round(f1_micro,4), round(f1_macro,4), round(f1_weighted,4)))

    # Label-wise f-scores.
    prec, rec, f1, supp = precision_recall_fscore_support(y_true, y_pred, \
        labels=classes, average=None)
    json_obj['labels'] = classes
    json_obj['prec'] = prec.tolist()
    json_obj['rec'] = rec.tolist()
    json_obj['f1'] = f1.tolist()
    json_obj['support'] = supp.tolist()

    # Prediction confidence and unambiguousness.
    if extra_scores:
        conf_mean = np.mean(confidence)
        conf_std = np.std(confidence)
        unambig_mean = np.mean(unambiguousness)
        unambig_std = np.std(unambiguousness)
        json_obj['confidence_mean'] = conf_mean
        json_obj['confidence_std'] = conf_std
        json_obj['unambiguousness_mean'] = unambig_mean
        json_obj['unambiguousness_std'] = unambig_std

        logger.info('Confidence for correct label [mean (std)]: %.4f (%.4f).' \
                    % (round(conf_mean, 4), round(conf_std, 4)))

        logger.info('Unambiguousness of the predictions (distance of best to ' \
                    + 'second-best prediction) [mean (std)]: %.4f (%.4f).' \
                    % (round(unambig_mean, 4), round(unambig_std, 4)))

    # Store results.
    if store:
        if not os.path.isdir(config.eval_dir):
            os.makedirs(config.eval_dir)
        filename = os.path.join(config.eval_dir, 'output_eval_'+tag+'.jsonl')

        with open(filename, 'a') as f:
            json_str = json.dumps(json_obj)
            f.write(json_str + '\n')
            f.flush()

        logger.debug('Appended output evaluations to %s.' % filename)

    return 1 - acc

if __name__ == '__main__':
    pass


