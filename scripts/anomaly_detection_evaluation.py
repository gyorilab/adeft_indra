import json
import numpy as np
from itertools import chain
from itertools import combinations
from multiprocessing import Lock
from multiprocessing import Pool
from collections import defaultdict
from sklearn.svm import OneClassSVM
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer

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
manager = ADResultsManager('anomaly_detection_evaluation')
reverse_model_map = {value: key for
                     key, value in available_shortforms.items()}


def nontrivial_subsets(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r)
                               for r in range(1, len(s)+1))


def evaluation_wrapper(args):
    model_name, nu, max_features, rng, n_jobs = args
    params = {'nu': nu, 'max_features': max_features}
    if manager.in_table(model_name, json.dumps(params)):
        with lock:
            print(f'Results for {model_name}, {params} have already'
                  ' been computed')
        return
    results = evaluate_anomaly_detection(model_name, nu, max_features, rng)
    manager.add_row([model_name, json.dumps(params), json.dumps(results)])


def evaluate_anomaly_detection(model_name, nu, max_features, rng, n_jobs):
    ad = load_disambiguator(reverse_model_map[model_name])
    model_name = escape_filename(':'.join(sorted(ad.shortforms)))
    params = {'nu': nu, 'max_features': max_features}

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
    grounded_entities = [g[0] for g in grounded_entities[0:6]]
    results = defaultdict(dict)
    for train_entities in nontrivial_subsets(grounded_entities):
        print(train_entities)
        baseline = [(text, label, pmid)
                    for text, label, pmid in corpus if label
                    in train_entities]
        if not baseline:
            continue
        texts, labels, pmids = zip(*baseline)
        texts, labels, pmids = list(texts), list(labels), list(pmids)
        baseline = None
        pipeline = Pipeline([('tfidf',
                              AdeftTfidfVectorizer(max_features=max_features,
                                                   stop_words=stop_words)),
                             ('forest_oc_svm',
                              ForestOneClassSVM(nu=nu,
                                                cache_size=1000,
                                                n_estimators=1000, n_jobs=1,
                                                random_state=rng))])
        anomalous = ((text, label, pmid) for text, label, pmid in corpus
                     if label not in train_entities)
        anomalous_texts, anomalous_labels, anomalous_pmids = zip(*anomalous)
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
        train_splits = StratifiedKFold(n_splits=5).split(texts, labels)
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
        folds_dict = {}
        sens_list = []
        spec_list = []
        j_list = []
        acc_list = []
        for i, (train, test) in enumerate(train_splits):
            pipeline.fit(Xin[train[0]:train[-1]+1],
                         true_labels_in[train[0]:train[-1]+1])
            preds_in = pipeline.predict(Xin[test[0]:test[-1]+1])
            preds_out = pipeline.predict(Xout)
            pipeline.estimator_ = None
            sens = \
                sensitivity_score(np.hstack((np.full(len(preds_in), 1.0),
                                            np.full(len(preds_out), -1.0))),
                                  np.hstack((preds_in, preds_out)),
                                  pos_label=-1)
            spec = \
                specificity_score(np.hstack((np.full(len(preds_in), 1.0),
                                             np.full(len(preds_out), -1.0))),
                                  np.hstack((preds_in, preds_out)),
                                  pos_label=-1)
            acc = \
                accuracy_score(np.hstack((np.full(len(preds_in), 1.0),
                                         np.full(len(preds_out), -1.0))),
                               np.hstack((preds_in, preds_out)))
            train_dict = {'pmids': pmids_in[train[0]:train[-1]+1].tolist(),
                          'true_label':
                          true_labels_in[train[0]:train[-1]+1].tolist()}
            test_dict = {'pmids': pmids_in[test[0]:test[-1]+1].tolist() +
                         pmids_out.tolist(),
                         'true_label':
                         true_labels_in[test[0]:test[-1]+1].tolist() +
                         true_labels_out.tolist(),
                         'prediction':  preds_in.tolist() + preds_out.tolist(),
                         'true': [1.0]*len(preds_in) + [-1.0]*len(preds_out)}
            scores_dict = {'specificity': spec, 'sensitivity': sens,
                           'youdens_j_score': spec + sens - 1,
                           'accuracy': acc}
            folds_dict[i] = {'training_data_info': train_dict,
                             'test_data_info': test_dict,
                             'scores': scores_dict}
            sens_list.append(sens)
            spec_list.append(spec)
            j_list.append(spec + sens - 1)
            acc_list.append(acc)
        mean_sens = np.mean(sens_list)
        mean_spec = np.mean(spec_list)
        mean_j = np.mean(j_list)
        mean_acc = np.mean(acc_list)
        std_sens = np.std(sens_list)
        std_spec = np.std(spec_list)
        std_j = np.std(j_list)
        std_acc = np.std(acc_list)
        aggregate_scores = {'sensitivity': {'mean': mean_sens,
                                            'std': std_sens},
                            'specificity': {'mean': mean_spec,
                                            'std': std_spec},
                            'youdens_j_score': {'mean': mean_j,
                                                'std': std_j},
                            'accuracy': {'mean': mean_acc,
                                         'std': std_acc}}
        results[json.dumps(train_entities)] = {'folds': folds_dict,
                                               'results': aggregate_scores,
                                               'params': params}
    return results


if __name__ == '__main__':

    model_data_counts = []
    for model_name in set(available_shortforms.values()):
        ad = load_disambiguator(reverse_model_map[model_name])
        stats = ad.classifier.stats
        total_count = sum(stats['label_distribution'].values)
        model_data_counts.append((model_name, total_count))
    model_data_counts.sort(key=lambda x: -x[1])

    batch1 = [x[0] for x in model_data_counts[50:]]
    batch2 = [x[0] for x in model_data_counts[5:50]]
    batch3 = [x[0] for x in model_data_counts[0:5]]
    main_rng = np.random.RandomState(1729)

    nu_list = [0.1, 0.2, 0.4]
    mf_list = [100, 1000, 10000]
    cases = []

    for model_name in batch1:
        for nu in nu_list:
            for mf in mf_list:
                rng = np.random.RandomState(main_rng.randint(2**32))
                cases.append((model_name, nu, mf, rng, 1))
    main_rng.shuffle(cases)

    with Pool(64) as pool:
        pool.map(evaluation_wrapper, cases, chunksize=1)

    for model_name in batch2:
        for nu in nu_list:
            for mf in mf_list:
                rng = np.random.RandomState(main_rng.randint(2**32))
                cases.append((model_name, nu, mf, rng, 2))
    main_rng.shuffle(cases)
    with Pool(32) as pool:
        pool.map(evaluation_wrapper, cases, chunksize=1)

    for model_name in batch2:
        for nu in nu_list:
            for mf in mf_list:
                rng = np.random.RandomState(main_rng.randint(2**32))
                cases.append((model_name, nu, mf, rng, 4))
    main_rng.shuffle(cases)
    with Pool(16) as pool:
        pool.map(evaluation_wrapper, cases, chunksize=1)
