import json
import math
import numpy as np
from itertools import chain
from itertools import combinations
from multiprocessing import Lock
from multiprocessing import Pool
from collections import defaultdict
from sklearn.svm import OneClassSVM
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

from adeft import available_shortforms
from adeft.nlp import english_stopwords
from adeft.modeling.label import AdeftLabeler
from adeft.disambiguate import load_disambiguator

from adeft_indra.tfidf import AdeftTfidfVectorizer
from adeft_indra.content import get_pmids_for_agent_text
from adeft_indra.content import get_plaintexts_for_pmids
from adeft_indra.model_building.escape import escape_filename
from adeft_indra.db.anomaly_detection import ADResultsManager
from adeft_indra.anomaly_detection.stats import sensitivity_score
from adeft_indra.anomaly_detection.stats import specificity_score
from adeft_indra.anomaly_detection.models import ForestOneClassSVM


lock = Lock()
manager = ADResultsManager('ad_nested_crossval')
reverse_model_map = {value: key for
                     key, value in available_shortforms.items()}


def nontrivial_subsets(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r)
                               for r in range(1, len(s)+1))


class SubsetSampler(object):
    def __init__(self, iterable, random_state):
        self.arr = list(iterable)
        self.sample_size = 0
        self.current_index = 0
        if isinstance(random_state, int):
            self.rng = np.random.RandomState(random_state)
        else:
            self.rng = random_state
        self.rng.shuffle(self.arr)

    def _reset(self):
        self.rng.shuffle(self.arr)
        self.current_index == 0

    def _sample_one(self):
        if self.current_index == len(self.arr):
            self._reset()
        result = self.arr[self.current_index]
        self.current_index += 1
        return result

    def sample(self, k):
        if k > len(self.arr)/2:
            res = set()
            while len(res) < len(self.arr) - k:
                res.add(self._sample_one())
            return [x for x in self.arr if x not in res]
        else:
            res = set()
            while len(res) < k:
                res.add(self._sample_one())
            return list(res)


class NestedKFold(object):
    def __init__(self, n_splits_outter=5, n_splits_inner=5):
        self.n_splits_outter = n_splits_outter
        self.n_splits_inner = n_splits_inner

    def split(self, X):
        outter_splitter = KFold(n_splits=self.n_splits_outter)
        inner_splitter = KFold(n_splits=self.n_splits_inner)
        outter_splits = outter_splitter.split(X)
        for outter_train, outter_test in outter_splits:
            inner_splits = inner_splitter.split(outter_train)
            inner = [(outter_train[inner_train], outter_train[inner_test])
                     for inner_train, inner_test in inner_splits]
            yield inner, outter_test


def evaluation_wrapper(args):
    model_name, nu, mf_a, mf_b, n_estimators, rng, n_jobs = args
    params = {'nu': nu, 'mf_a': mf_a, 'mf_b': mf_b,
              'n_estimators': n_estimators}
    if manager.in_table(model_name, json.dumps(params)):
        with lock:
            print(f'Results for {model_name}, {params} have already'
                  ' been computed')
        return
    results = evaluate_anomaly_detection(model_name, nu, mf_a, mf_b,
                                         n_estimators, rng,
                                         n_jobs)
    if results:
        manager.add_row([model_name, json.dumps(params), json.dumps(results)])
        with lock:
            print(f'Success for {nu}, {mf_a}, {mf_b}, {model_name}')
    else:
        with lock:
            print(f'Problem with {nu}, {mf_a}, {mf_b}, {model_name}')


def evaluate_anomaly_detection(model_name, nu, mf_a, mf_b,
                               n_estimators, rng, n_jobs):
    ad = load_disambiguator(reverse_model_map[model_name])
    model_name = escape_filename(':'.join(sorted(ad.shortforms)))
    params = {'nu': nu, 'mf_a': mf_a, 'mf_b': mf_b,
              'n_estimators': n_estimators}
    with lock:
        print(f'Working on {model_name} with params {params}')
    stop_words = set(english_stopwords) | set(ad.shortforms)

    all_pmids = set()
    for shortform in ad.shortforms:
        all_pmids.update(get_pmids_for_agent_text(shortform))
    text_dict = get_plaintexts_for_pmids(all_pmids,
                                         contains=ad.shortforms)
    labeler = AdeftLabeler(ad.grounding_dict)
    corpus = labeler.build_from_texts((text, pmid) for pmid, text
                                      in text_dict.items())
    text_dict = None
    entity_pmid_map = defaultdict(list)
    for text, label, pmid in corpus:
        entity_pmid_map[label].append(pmid)
    grounded_entities = [(entity, len(entity_pmid_map[entity]))
                         for entity in entity_pmid_map
                         if entity != 'ungrounded']
    grounded_entities = [(entity, length) for entity, length in
                         grounded_entities if length >= 10]
    grounded_entities.sort(key=lambda x: -x[1])
    grounded_entities = [g[0] for g in grounded_entities[0:5]]
    sampled_subsets = []
    for k in range(1, len(grounded_entities)):
        sampler = SubsetSampler(grounded_entities, rng)
        n_samples = len(grounded_entities)
        sampled_subsets.extend([sampler.sample(k) for _ in range(n_samples)])
    rng.shuffle(sampled_subsets)
    results_dict = {}
    for train_entities in sampled_subsets:
        num_classes = len(train_entities)
        print(train_entities)
        baseline = [(text, label, pmid)
                    for text, label, pmid in corpus if label
                    in train_entities]
        if not baseline or len(baseline) < 25:
            continue
        texts, labels, pmids = zip(*baseline)
        texts, labels, pmids = list(texts), list(labels), list(pmids)
        baseline = None
        max_features = mf_a + mf_b * (num_classes-1)
        pipeline = Pipeline([('tfidf',
                              AdeftTfidfVectorizer(max_features=max_features,
                                                   stop_words=stop_words)),
                             ('forest_oc_svm',
                              ForestOneClassSVM(nu=nu,
                                                cache_size=1000,
                                                n_estimators=n_estimators,
                                                n_jobs=n_jobs,
                                                random_state=rng))])
        # pipeline = Pipeline([('tfidf',
        #                       AdeftTfidfVectorizer(max_features=max_features,
        #                                            stop_words=stop_words)),
        #                      ('oc_svm',
        #                       OneClassSVM(nu=nu, kernel='linear'))])
        anomalous = ((text, label, pmid) for text, label, pmid in corpus
                     if label not in train_entities)

        try:
            anomalous_texts, anomalous_labels, anomalous_pmids = zip(*anomalous)
        except ValueError:
            print
            continue
        anomalous_texts = list(anomalous_texts)
        anomalous_labels = list(anomalous_labels)
        anomalous_pmids = list(anomalous_pmids)
        rng2 = np.random.RandomState(561)
        rng2.shuffle(texts)
        rng2 = np.random.RandomState(561)
        rng2.shuffle(labels)
        rng2 = np.random.RandomState(561)
        rng2.shuffle(pmids)
        rng2 = None
        nested_splitter = NestedKFold(n_splits_outter=5, n_splits_inner=5)
        nested_splits = nested_splitter.split(texts)
        Xin = np.array(texts)
        Xout = np.array(anomalous_texts)
        texts = None
        anomalous_texts = None
        true_labels_in = np.array(labels)
        true_labels_out = np.array(anomalous_labels)
        pmids_in = np.array(pmids)
        pmids_out = np.array(anomalous_pmids)
        labels = None
        anomalous_labels = None
        pmids = None
        anomalous_pmids = None
        inner_folds_dict = defaultdict(dict)
        outter_folds_dict = {}
        outter_spec_list = []
        for i, (inner_splits, outter_test) in enumerate(nested_splits):
            inner_spec_list = []
            for j, (train, inner_test) in enumerate(inner_splits):
                pipeline.fit(Xin[train], true_labels_in[train])
                preds_in = pipeline.predict(Xin[inner_test])
                pipeline.estimator_ = None
                spec = \
                    specificity_score(np.full(len(preds_in), 1.0),
                                      preds_in,
                                      pos_label=-1)
                train_dict = {'pmids': pmids_in[train].tolist(),
                              'true_label':
                              true_labels_in[train].tolist()}
                test_dict = {'pmids':
                             pmids_in[inner_test].tolist(),
                             'true_label':
                             true_labels_in[inner_test].tolist(),
                             'prediction':  preds_in.tolist(),
                             'true': [1.0]*len(preds_in)}
                inner_folds_dict[i][j] = {'training_data_info': train_dict,
                                          'test_data_info': test_dict,
                                          'spec': spec}
                inner_spec_list.append(spec)
            mean_spec = np.mean(inner_spec_list)
            std_spec = np.std(inner_spec_list)
            outter_train = np.concatenate([x for split in inner_splits
                                           for x in split])
            pipeline.fit(Xin[outter_train], true_labels_in[outter_train])
            preds_in = pipeline.predict(Xin[outter_test])
            pipeline.estimator_ = None
            spec = \
                specificity_score(np.full(len(preds_in), 1.0),
                                  preds_in,
                                  pos_label=-1)
            train_dict = {'pmids': pmids_in[outter_train].tolist(),
                          'true_label':
                          true_labels_in[outter_train].tolist()}
            test_dict = {'pmids':
                         pmids_in[outter_test].tolist(),
                         'true_label':
                         true_labels_in[outter_test].tolist(),
                         'prediction':  preds_in.tolist(),
                         'true': [1.0]*len(preds_in)}
            outter_folds_dict[i] = {'training_data_info': train_dict,
                                    'test_data_info': test_dict,
                                    'spec': spec,
                                    'mean_inner_spec': mean_spec,
                                    'std_inner_spec': std_spec}
            outter_spec_list.append(spec)
        mean_spec = np.mean(outter_spec_list)
        std_spec = np.std(outter_spec_list)
        pipeline.fit(Xin, true_labels_in)
        preds_out = pipeline.predict(Xout)
        pipeline.estimator_ = None
        sens = sensitivity_score(np.full(len(preds_out), -1.0),
                                 preds_out,
                                 pos_label=-1)
        train_dict = {'pmids': pmids_in.tolist(),
                      'true_label':
                      true_labels_in.tolist()}
        test_dict = {'pmids':
                     pmids_out.tolist(),
                     'true_label':
                     true_labels_out.tolist(),
                     'prediction':  preds_out.tolist(),
                     'true': [-1.0]*len(preds_out)}
        outter_dict = {'training_data_info': train_dict,
                       'test_data_info': test_dict,
                       'sens': sens,
                       'mean_outter_spec': mean_spec,
                       'std_outter_spec': std_spec}
        results_dict[json.dumps(train_entities)] = {'inner_folds_dict':
                                                    inner_folds_dict,
                                                    'outter_folds_dict':
                                                    outter_folds_dict,
                                                    'outter_dict':
                                                    outter_dict}
    return results_dict


if __name__ == '__main__':
    model_data_counts = []
    for model_name in set(available_shortforms.values()):
        ad = load_disambiguator(reverse_model_map[model_name])
        stats = ad.classifier.stats
        total_count = sum(stats['label_distribution'].values())
        model_data_counts.append((model_name, total_count))
    model_data_counts.sort(key=lambda x: -x[1])

    batch1 = [x[0] for x in model_data_counts[10:]]
    batch2 = [x[0] for x in model_data_counts[5:10]]
    batch3 = [x[0] for x in model_data_counts[0:5]]
    main_rng = np.random.RandomState(1729)

    rows = manager.get_results()
    nu = 0.225
    mf_a = 5
    mf_b = 30
    n_est = 1000
    cases = []
    for model_name in batch2:
        rng = np.random.RandomState(main_rng.randint(2**32))
        cases.append((model_name, nu, mf_a, mf_b, n_est, rng,
                      48))
    main_rng.shuffle(cases)

    with Pool(1) as pool:
        pool.map(evaluation_wrapper, cases, chunksize=1)
    # for model_name in batch2:
    #     for nu in nu_list:
    #         for mf in mf_list:
    #             rng = np.random.RandomState(main_rng.randint(2**32))
    #             cases.append((model_name, nu, mf, rng, 2))
    # main_rng.shuffle(cases)
    # with Pool(32) as pool:
    #     pool.map(evaluation_wrapper, cases, chunksize=1)

    # for model_name in batch2:
    #     for nu in nu_list:
    #         for mf in mf_list:
    #             rng = np.random.RandomState(main_rng.randint(2**32))
    #             cases.append((model_name, nu, mf, rng, 4))
    # main_rng.shuffle(cases)
    # with Pool(16) as pool:
    #     pool.map(evaluation_wrapper, cases, chunksize=1)
