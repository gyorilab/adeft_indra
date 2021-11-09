import json
import math
import logging
import numpy as np
from itertools import chain
from itertools import combinations
from multiprocessing import Lock
from multiprocessing import Pool
from collections import defaultdict
from sklearn.svm import OneClassSVM
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from adeft import available_shortforms
from adeft.nlp import english_stopwords
from adeft.modeling.label import AdeftLabeler
from adeft.disambiguate import load_disambiguator

from adeft_indra.tfidf import AdeftTfidfVectorizer
from adeft_indra.content import get_pmids_for_agent_text
from adeft_indra.content import get_plaintexts_for_pmids
from adeft_indra.model_building.escape import escape_filename
from adeft_indra.db.anomaly_detection import ADResultsManager

from opaque.stats import sensitivity_score
from opaque.stats import specificity_score
from opaque.ood.forest_svm import AnyMethodPipeline
from opaque.ood.forest_svm import ForestOneClassSVM

lock = Lock()
manager = ADResultsManager("nested_crossvalidation6")
reverse_model_map = {value: key for key, value in available_shortforms.items()}


def nontrivial_subsets(iterable):
    s = list(iterable)
    return chain.from_iterable(
        combinations(s, r) for r in range(1, len(s) + 1)
    )


class SubsetSampler(object):
    def __init__(self, iterable, random_state):
        self.arr = list(iterable)
        self.current_index = 0
        if isinstance(random_state, int):
            self.rng = np.random.RandomState(random_state)
        else:
            self.rng = random_state
        self.rng.shuffle(self.arr)

    def _reset(self):
        self.rng.shuffle(self.arr)
        self.current_index = 0

    def sample(self, k):
        if self.current_index + k >= len(self.arr):
            self._reset()
        res = self.arr[self.current_index:self.current_index + k]
        self.current_index += k
        return sorted(res)

    def multiple_samples(self, n, k):
        """Draw n samples each of length k without replacement."""
        results = set()
        for _ in range(n):
            results.add(tuple(self.sample(k)))
        return list(results)


class NestedKFold(object):
    def __init__(self, n_splits_outter=5, n_splits_inner=5):
        self.n_splits_outter = n_splits_outter
        self.n_splits_inner = n_splits_inner

    def split(self, X, y):
        outter_splitter = StratifiedKFold(n_splits=self.n_splits_outter)
        inner_splitter = StratifiedKFold(n_splits=self.n_splits_inner)
        outter_splits = outter_splitter.split(X, y)
        for outter_train, outter_test in outter_splits:
            inner_splits = inner_splitter.split(X[outter_train],
                                                y[outter_train])
            inner = [
                (outter_train[inner_train], outter_train[inner_test])
                for inner_train, inner_test in inner_splits
            ]
            yield inner, outter_test


def evaluation_wrapper(args):
    model_name, nu, max_features, n_estimators, rng, n_jobs = args
    params = {
        "nu": nu,
        "max_features": max_features,
        "n_estimators": n_estimators
    }
    if manager.in_table(model_name, json.dumps(params)):
        with lock:
            print(
                f"Results for {model_name}, {params} have already"
                " been computed"
            )
        return
    results = evaluate_anomaly_detection(
        model_name, nu, max_features, n_estimators, rng, n_jobs
    )
    if results:
        manager.add_row([model_name, json.dumps(params), json.dumps(results)])
        with lock:
            print(f"Success for {nu}, {max_features}, {model_name}")
    else:
        with lock:
            print(f"Problem with {nu}, {max_features}, {model_name}")


def evaluate_anomaly_detection(
        model_name, nu, max_features, n_estimators, rng, n_jobs
):
    ad = load_disambiguator(reverse_model_map[model_name])
    params = {
        "nu": nu,
        "max_features": max_features,
        "n_estimators": n_estimators
    }
    with lock:
        print(f"Working on {model_name} with params {params}")
    stop_words = set(english_stopwords) | set(ad.shortforms)

    all_pmids = set()
    for shortform in ad.shortforms:
        all_pmids.update(get_pmids_for_agent_text(shortform))
    text_dict = get_plaintexts_for_pmids(all_pmids, contains=ad.shortforms)
    labeler = AdeftLabeler(ad.grounding_dict)
    corpus = labeler.build_from_texts(
        (text, pmid) for pmid, text in text_dict.items()
    )
    text_dict = None
    entity_pmid_map = defaultdict(list)
    for text, label, pmid in corpus:
        entity_pmid_map[label].append(pmid)
    grounded_entities = [
        (entity, len(entity_pmid_map[entity]))
        for entity in entity_pmid_map
        if entity != "ungrounded"
    ]
    grounded_entities = [
        (entity, length) for entity, length in grounded_entities
        if length >= 10
    ]
    grounded_entities.sort(key=lambda x: -x[1])
    grounded_entities = [g[0] for g in grounded_entities[0:9]]
    sampled_subsets = []
    for k in range(6, min(len(grounded_entities), 8)):
        sampler = SubsetSampler(grounded_entities, rng)
        n_samples = len(grounded_entities)
        sampled_subsets.extend(sampler.multiple_samples(n_samples, k))
    rng.shuffle(sampled_subsets)
    results_dict = {}
    for train_entities in sampled_subsets:
        print(5, train_entities)
        baseline = [
            (text, label, pmid)
            for text, label, pmid in corpus
            if label in train_entities
        ]
        if not baseline or len(baseline) < 25:
            continue
        texts, labels, pmids = zip(*baseline)
        texts, labels, pmids = list(texts), list(labels), list(pmids)
        baseline = None
        pipeline = AnyMethodPipeline(
            [
                (
                    "tfidf",
                    AdeftTfidfVectorizer(
                        max_features=max_features, stop_words=stop_words
                    ),
                ),
                (
                    "forest_oc_svm",
                    ForestOneClassSVM(
                        RandomForestClassifier(
                            n_estimators=n_estimators,
                            n_jobs=n_jobs,
                            random_state=rng
                        ),
                        nu=nu,
                        cache_size=1000,

                    ),
                ),
            ]
        )
        # pipeline = Pipeline([('tfidf',
        #                       AdeftTfidfVectorizer(max_features=max_features,
        #                                            stop_words=stop_words)),
        #                      ('oc_svm',
        #                       OneClassSVM(nu=nu, kernel='linear'))])
        anomalous = (
            (text, label, pmid)
            for text, label, pmid in corpus
            if label not in train_entities
        )

        try:
            anom_texts, anom_labels, anom_pmids = zip(*anomalous)
        except ValueError:
            continue
        anom_texts = list(anom_texts)
        anom_labels = list(anom_labels)
        anom_pmids = list(anom_pmids)
        rng2 = np.random.RandomState(561)
        rng2.shuffle(texts)
        rng2 = np.random.RandomState(561)
        rng2.shuffle(labels)
        rng2 = np.random.RandomState(561)
        rng2.shuffle(pmids)
        rng2 = None
        Xin = np.array(texts)
        Xout = np.array(anom_texts)
        texts = None
        anom_texts = None
        true_labels_in = np.array(labels)
        true_labels_out = np.array(anom_labels)
        pmids_in = np.array(pmids)
        pmids_out = np.array(anom_pmids)
        labels = None
        anom_labels = None
        pmids = None
        anom_pmids = None
        nested_splitter = NestedKFold(n_splits_outter=5, n_splits_inner=5)
        nested_splits = nested_splitter.split(Xin, true_labels_in)
        inner_folds_dict = defaultdict(dict)
        outter_folds_dict = {}
        outter_spec_list = []
        for i, (inner_splits, outter_test) in enumerate(nested_splits):
            inner_spec_list = []
            for j, (train, inner_test) in enumerate(inner_splits):
                pipeline.fit(Xin[train], true_labels_in[train])
                ood_preds_in = pipeline.apply_method(
                    'ood_predict', Xin[inner_test]
                )
                forest_preds_in = pipeline.apply_method(
                    'forest_predict', Xin[inner_test]
                    )
                forest_probs_in = pipeline.apply_method(
                    'forest_predict_proba', Xin[inner_test]
                )
                pipeline.estimator_ = None
                spec = specificity_score(
                    np.full(len(ood_preds_in), 1.0), ood_preds_in, pos_label=-1
                )
                bacc = balanced_accuracy_score(
                    true_labels_in[inner_test],
                    forest_preds_in
                )
                train_dict = {
                    "pmids": pmids_in[train].tolist(),
                    "true_label": true_labels_in[train].tolist(),
                    "classes": pipeline.named_steps['forest_oc_svm'].
                    forest.classes_.tolist()
                }
                test_dict = {
                    "pmids": pmids_in[inner_test].tolist(),
                    "true_label": true_labels_in[inner_test].tolist(),
                    "ood_prediction": ood_preds_in.tolist(),
                    "forest_prediction": forest_preds_in.tolist(),
                    "forest_probs": forest_probs_in.tolist(),
                    "true": [1.0] * len(ood_preds_in),
                }
                inner_folds_dict[i][j] = {
                    "training_data_info": train_dict,
                    "test_data_info": test_dict,
                    "spec": spec,
                    "bacc": bacc
                }
                inner_spec_list.append(spec)
            mean_spec = np.mean(inner_spec_list)
            std_spec = np.std(inner_spec_list)
            outter_train = np.concatenate(
                [x for split in inner_splits for x in split]
            )
            pipeline.fit(Xin[outter_train], true_labels_in[outter_train])
            ood_preds_in = pipeline.apply_method(
                'ood_predict', Xin[outter_test]
            )
            forest_preds_in = pipeline.apply_method(
                'forest_predict', Xin[outter_test]
            )
            forest_probs_in = pipeline.apply_method(
                'forest_predict_proba', Xin[outter_test]
            )
            pipeline.estimator_ = None
            spec = specificity_score(
                np.full(len(ood_preds_in), 1.0), ood_preds_in, pos_label=-1
            )
            bacc = balanced_accuracy_score(
                true_labels_in[outter_test],
                forest_preds_in
            )
            train_dict = {
                "pmids": pmids_in[outter_train].tolist(),
                "true_label": true_labels_in[outter_train].tolist(),
                "classes": pipeline.named_steps['forest_oc_svm'].
                forest.classes_.tolist()
            }
            test_dict = {
                "pmids": pmids_in[outter_test].tolist(),
                "true_label": true_labels_in[outter_test].tolist(),
                "prediction": ood_preds_in.tolist(),
                "forest_prediction": forest_preds_in.tolist(),
                "forest_probs": forest_probs_in.tolist(),
                "true": [1.0] * len(ood_preds_in),
            }
            outter_folds_dict[i] = {
                "training_data_info": train_dict,
                "test_data_info": test_dict,
                "spec": spec,
                "bacc": bacc,
                "mean_inner_spec": mean_spec,
                "std_inner_spec": std_spec,
            }
            outter_spec_list.append(spec)
        mean_spec = np.mean(outter_spec_list)
        std_spec = np.std(outter_spec_list)
        pipeline.fit(Xin, true_labels_in)
        preds_out = pipeline.apply_method(
            'ood_predict', Xout
        )
        forest_probs_out = pipeline.apply_method(
            'forest_predict_proba', Xout
        )
        pipeline.estimator_ = None
        sens = sensitivity_score(
            np.full(len(preds_out), -1.0), preds_out, pos_label=-1
        )
        train_dict = {
            "pmids": pmids_in.tolist(),
            "true_label": true_labels_in.tolist(),
            "classes": pipeline.named_steps['forest_oc_svm'].
            forest.classes_.tolist()
        }
        test_dict = {
            "pmids": pmids_out.tolist(),
            "true_label": true_labels_out.tolist(),
            "prediction": preds_out.tolist(),
            "forest_probs": forest_probs_out.tolist(),
            "true": [-1.0] * len(preds_out),
        }
        outter_dict = {
            "training_data_info": train_dict,
            "test_data_info": test_dict,
            "sens": sens,
            "mean_outter_spec": mean_spec,
            "std_outter_spec": std_spec,
        }
        results_dict[json.dumps(train_entities)] = {
            "inner_folds_dict": inner_folds_dict,
            "outter_folds_dict": outter_folds_dict,
            "outter_dict": outter_dict,
        }
    return results_dict


if __name__ == "__main__":
    model_data_counts = []
    for model_name in set(available_shortforms.values()):
        ad = load_disambiguator(reverse_model_map[model_name])
        stats = ad.classifier.stats
        total_count = sum(stats["label_distribution"].values())
        model_data_counts.append((model_name, total_count))
    model_data_counts.sort(key=lambda x: -x[1])

    batch0 = [x[0] for x in model_data_counts[50:]]
    batch1 = [x[0] for x in model_data_counts[10:]]
    batch2 = [x[0] for x in model_data_counts[5:10]]
    batch3 = [x[0] for x in model_data_counts[0:5]]
    main_rng = np.random.RandomState(1729)

    nu_list = [0.2375]
    mf_list = [50]
    n_est = 1000
    cases = []

    for model_name in batch1:
        for nu in nu_list:
            for mf in mf_list:
                rng = np.random.RandomState(main_rng.randint(2 ** 32))
                cases.append((model_name, nu, mf, n_est, rng, 1))
    main_rng.shuffle(cases)


with Pool(32) as pool:
    pool.map(evaluation_wrapper, cases, chunksize=1)
