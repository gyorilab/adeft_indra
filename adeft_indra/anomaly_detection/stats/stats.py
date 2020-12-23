import logging
from multiprocessing import Pool

import numpy as np
import scipy as sp
from scipy.stats import beta
from scipy.optimize import brentq
from scipy.special import loggamma

from ._stats import prevalence_credible_interval_exact


logger = logging.getLogger(__file__)


def true_positives(y_true, y_pred, pos_label=1):
    return sum(1 if expected == pos_label and predicted == pos_label else 0
               for expected, predicted in zip(y_true, y_pred))


def true_negatives(y_true, y_pred, pos_label=1):
    return sum(1 if expected != pos_label and predicted != pos_label else 0
               for expected, predicted in zip(y_true, y_pred))


def false_positives(y_true, y_pred, pos_label=1):
    return sum(1 if expected != pos_label and predicted == pos_label else 0
               for expected, predicted in zip(y_true, y_pred))


def false_negatives(y_true, y_pred, pos_label=1):
    return sum(1 if expected == pos_label and predicted != pos_label else 0
               for expected, predicted in zip(y_true, y_pred))


def sensitivity_score(y_true, y_pred, pos_label=1):
    tp = true_positives(y_true, y_pred, pos_label)
    fn = false_negatives(y_true, y_pred, pos_label)
    try:
        result = tp/(tp + fn)
    except ZeroDivisionError:
        logger.warning('No positive examples in sample. Returning 0.0')
        result = 0.0
    return result


def specificity_score(y_true, y_pred, pos_label=1):
    tn = true_negatives(y_true, y_pred, pos_label)
    fp = false_positives(y_true, y_pred, pos_label)
    try:
        result = tn/(tn + fp)
    except ZeroDivisionError:
        logger.warning('No negative examples in sample. Returning 0.0')
        result = 0.0
    return result


def youdens_j_score(y_true, y_pred, pos_label=1):
    sens = sensitivity_score(y_true, y_pred, pos_label)
    spec = specificity_score(y_true, y_pred, pos_label)
    return sens + spec - 1


def sample_interval(n, t, sens_shape, sens_range,
                    spec_shape, spec_range, alpha):
    sp.random.seed()

    scale = sens_range[1] - sens_range[0]
    loc = sens_range[0]
    sens = beta.rvs(sens_shape[0], sens_shape[1], scale=scale, loc=loc)
    scale = spec_range[1] - spec_range[0]
    loc = spec_range[0]
    spec = beta.rvs(spec_shape[0], spec_shape[1], scale=scale, loc=loc)
    return prevalence_credible_interval_exact(n, t, sens, spec, alpha)


def prevalence_credible_interval(n, t, sens_shape, sens_range,
                                 spec_shape, spec_range, alpha,
                                 num_samples=5000, n_jobs=1):
    if n_jobs > 1:
        with Pool(n_jobs) as pool:
            future_results = [pool.apply_async(sample_interval,
                                               args=(n, t, sens_shape,
                                                     sens_range,
                                                     spec_shape,
                                                     spec_range,
                                                     alpha))
                              for i in range(num_samples)]
            results = [interval.get() for interval in future_results]
    else:
        results = [sample_interval(n, t, sens_shape, sens_range,
                                   spec_shape, spec_range, alpha)
                   for i in range(num_samples)]
    return (np.mean([t[0] for t in results]),
            np.mean([t[1] for t in results]))
