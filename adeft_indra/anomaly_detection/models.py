import logging

import numpy as np


from sklearn.svm import OneClassSVM
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from adeft.nlp import english_stopwords


from adeft_indra.tfidf import AdeftTfidfVectorizer, FrozenTfidfVectorizer
from adeft_indra.anomaly_detection.tree_kernel import make_random_forest_kernel
from .stats import sensitivity_score, specificity_score, youdens_j_score, \
    make_anomaly_detector_scorer

logger = logging.getLogger(__file__)

gensim_logger = logging.getLogger('gensim')
sklearn_logger = logging.getLogger('sklearn')
gensim_logger.setLevel('WARNING')
sklearn_logger.setLevel('WARNING')


class BaseAnomalyDetector(object):
    def __init__(self, tfidf_path=None, blacklist=None):
        self.tfidf_path = tfidf_path
        self.blacklist = [] if blacklist is None else blacklist
        self.estimator = None
        self.sensitivity = None
        self.specificity = None
        self.best_score = None
        self.best_params = None
        self.cv_results = None
        # Terms in the blacklist are tokenized and preprocessed just as in
        # the underlying model and then added to the set of excluded stop words
        tokenize = TfidfVectorizer().build_tokenizer()
        tokens = [token.lower() for token in
                  tokenize(' '.join(self.blacklist))]
        self.stop = set(english_stopwords).union(tokens)
        self.__param_mapping = self._get_param_mapping()
        self.__inverse_param_mapping = {value: key for
                                        key, value in
                                        self.__param_mapping.items()}

    def train(self, texts, **params):
        raise NotImplementedError

    def cv(self, texts, anomalous_texts, param_grid, n_jobs=1, cv=5,
           random_state=None):
        pipeline = self._make_pipeline()
        texts = list(texts)
        anomalous_texts = list(anomalous_texts)
        # Create crossvalidation splits for both the training texts and
        # the anomalous texts
        train_splits = KFold(n_splits=cv, random_state=random_state,
                             shuffle=True).split(texts)
        # Handle case where an insufficient amount of anomalous texts
        # are provided. In this case only specificity can be estimated
        if len(anomalous_texts) < cv:
            X = texts
            y = [1.0]*len(texts)
            splits = train_splits
        else:
            anomalous_splits = KFold(n_splits=cv, random_state=random_state,
                                     shuffle=True).split(anomalous_texts)
            # Combine training texts and anomalous texts into a single dataset
            # Give label -1.0 for anomalous texts, 1.0 otherwise
            X = texts + anomalous_texts
            y = [1.0]*len(texts) + [-1.0]*len(anomalous_texts)
            # Generate splits where training folds only contain training texts,
            # and test folds also contain both training and anomalous texts
            splits = ((train, np.concatenate((test,
                                              anom_test + len(texts))))
                      for (train, test), (_, anom_test)
                      in zip(train_splits, anomalous_splits))
        scorer = make_anomaly_detector_scorer()
        sensitivity_scorer = make_scorer(sensitivity_score, pos_label=-1.0)
        specificity_scorer = make_scorer(specificity_score, pos_label=-1.0)
        yj_scorer = make_scorer(youdens_j_score, pos_label=-1.0)
        scorer = {'sens': sensitivity_scorer, 'spec': specificity_scorer,
                  'yj': yj_scorer}

        param_grid = {self.__param_mapping[key]: value
                      for key, value in param_grid.items()}
        grid_search = GridSearchCV(pipeline, param_grid, scoring=scorer,
                                   n_jobs=n_jobs, cv=splits, refit=False,
                                   iid=False)
        grid_search.fit(X, y)
        cv_results = grid_search.cv_results_
        info = self._get_info(cv_results)
        sensitivity, specificity = info['sens_mean'], info['spec_mean']
        std_sensitivity, std_specificity = info['sens_std'], info['spec_std']
        params = info['params']
        best_score = sensitivity + specificity - 1
        logger.info('Best score of %s found for'
                    ' parameter values:\n%s' % (best_score,
                                                params))

        self.sensitivity = sensitivity
        self.specificity = specificity
        self.std_sensitivity = std_sensitivity
        self.std_specificity = std_specificity
        self.best_score = best_score
        self.best_params = params
        self.cv_results = cv_results
        self.train(texts, **params)

    def feature_importances(self):
        """Return list of n-gram features along with their SVM coefficients

        Returns
        -------
        list of tuple
            List of tuples with first element an n-gram feature and second
            element an SVM coefficient. Sorted by coefficient in decreasing
            order. Since this is a one class svm, all coefficients are
            positive.
        """
        if not self.estimator or not hasattr(self.estimator, 'named_steps') \
           or not hasattr(self.estimator.named_steps['oc_svm'], 'coef_'):
            raise RuntimeError('Classifier has not been fit')
        tfidf = self.estimator.named_steps['tfidf']
        classifier = self.estimator.named_steps['oc_svm']
        feature_names = tfidf.get_feature_names()
        coefficients = classifier.coef_.toarray().ravel()
        return sorted(zip(feature_names, coefficients),
                      key=lambda x: -x[1])

    def predict(self, texts):
        """Return list of predictions for a list of texts

        Parameters
        ----------
        texts : str

        Returns
        -------
        list of float
            Predicted labels for each text. 1.0 for anomalous, 0.0 otherwise
        """
        preds = self.estimator.predict(texts)
        return np.where(preds == -1.0, 1.0, 0.0)

    def confidence_interval(self, texts):
        pass

    def _get_info(self, cv_results):
        best_index = max(range(len(cv_results['mean_test_yj'])),
                         key=lambda i: cv_results['mean_test_yj'][i])
        sens = cv_results['mean_test_sens'][best_index]
        sens_std = cv_results['std_test_sens'][best_index]
        spec = cv_results['mean_test_spec'][best_index]
        spec_std = cv_results['std_test_spec'][best_index]
        params = cv_results['params'][best_index]
        params = {self.__inverse_param_mapping[key]: value
                  for key, value in params.items()}
        return {'sens_mean': sens, 'sens_std': sens_std,
                'spec_mean': spec, 'spec_std': spec_std,
                'params': params}

    def _make_scorer(self):
        sensitivity_scorer = make_scorer(sensitivity_score, pos_label=-1.0)
        specificity_scorer = make_scorer(specificity_score, pos_label=-1.0)
        yj_scorer = make_scorer(youdens_j_score, pos_label=-1.0)
        scorer = {'sens': sensitivity_scorer, 'spec': specificity_scorer,
                  'yj': yj_scorer}
        return scorer

    def _get_param_mapping(self):
        raise NotImplementedError

    def _make_pipeline(self, **params):
        raise NotImplementedError

    def _train(self, texts, **params):
        pipeline = self._make_pipeline(**params)
        pipeline.fit(texts)
        self.estimator = pipeline
        return self.estimator


class AdeftAnomalyDetector(BaseAnomalyDetector):
    def train(self, texts, nu=0.5, ngram_range=(1, 1), max_features=1000):
        params = {'nu': nu, 'ngram_range': ngram_range,
                  'max_features': max_features}
        return self._train(texts, **params)

    def _get_param_mapping(self):
        return {'nu': 'oc_svm__nu',
                'max_features': 'tfidf__max_features'}

    def _make_pipeline(self, nu=0.5, ngram_range=(1, 1), max_features=1000):
        return Pipeline([('tfidf',
                          AdeftTfidfVectorizer(dict_path=self.tfidf_path,
                                               max_features=max_features,
                                               stop_words=self.stop)),
                         ('oc_svm',
                          OneClassSVM(kernel='linear', nu=nu))])


class ForestOneClassSVM(BaseEstimator):
    def __init__(self,
                 # RandomForestClassifier parameters
                 # OneClassSVM parameters
                 tol=1e-3, nu=0.5, shrinking=True,
                 cache_size=200, verbose=False, max_iter=-1, **forest_params):

        self.tol = tol
        self.nu = nu
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter
        self.forest_params = forest_params
        self.estimator_ = None

    def fit(self, X, y, sample_weight=None, **params):
        num_classes = len(set(y))
        if num_classes > 1:
            forest_estimator = RandomForestClassifier(**self.forest_params)
            forest_estimator.fit(X, y, sample_weight=sample_weight)
            kernel = make_random_forest_kernel(forest_estimator)
        else:
            kernel = 'linear'
        estimator = OneClassSVM(kernel=kernel,
                                tol=self.tol,
                                nu=self.nu,
                                shrinking=self.shrinking,
                                cache_size=self.cache_size,
                                verbose=self.verbose,
                                max_iter=self.max_iter)
        estimator.fit(X)
        self.estimator_ = estimator
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def decision_function(self, X):
        return self.estimator_.decision_function(X)
