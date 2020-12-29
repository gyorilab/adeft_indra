import logging
from sklearn.metrics import make_scorer


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


def make_anomaly_detector_scorer():
    sensitivity_scorer = make_scorer(sensitivity_score, pos_label=-1.0)
    specificity_scorer = make_scorer(specificity_score, pos_label=-1.0)
    yj_scorer = make_scorer(youdens_j_score, pos_label=-1.0)
    scorer = {'sens': sensitivity_scorer, 'spec': specificity_scorer,
              'yj': yj_scorer}
    return scorer
