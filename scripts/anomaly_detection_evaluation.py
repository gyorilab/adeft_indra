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
    model_name, nu, max_features, rng = args
    params = {'nu': nu, 'max_features': max_features}
    if manager.in_table(model_name, json.dumps(params)):
        with lock:
            print(f'Results for {model_name}, {params} have already'
                  ' been computed')
        return
    results = evaluate_anomaly_detection(model_name, nu, max_features, rng)
    manager.add_row([model_name, json.dumps(params), json.dumps(results)])


def evaluate_anomaly_detection(model_name, nu, max_features, rng):
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
        baseline = [(text, label, pmid)
                    for text, label, pmid in corpus if label
                    in train_entities]
        if not baseline:
            continue
        texts, labels, pmids = zip(*baseline)
        pipeline = Pipeline([('tfidf',
                              AdeftTfidfVectorizer(max_features=max_features,
                                                   stop_words=stop_words)),
                             ('forest_oc_svm',
                              ForestOneClassSVM(nu=nu,
                                                cache_size=1000,
                                                n_estimators=1000, n_jobs=8,
                                                random_state=rng))])
        anomalous = [(text, label, pmid) for text, label, pmid in corpus
                     if label not in train_entities]
        anomalous_texts, anomalous_labels, anomalous_pmids = zip(*anomalous)
        train_splits = StratifiedKFold(n_splits=5, random_state=561,
                                       shuffle=True).split(texts, labels)
        splits = ((train, np.concatenate((test,
                                          np.arange(len(texts),
                                                    len(texts) +
                                                    len(anomalous_texts)))))
                  for train, test in train_splits)
        X = np.array(texts + anomalous_texts)
        true_labels = np.array(labels + anomalous_labels)
        all_pmids = np.array(pmids + anomalous_pmids)
        y = np.array([1.0]*len(texts) + [-1.0]*len(anomalous_texts))
        folds_dict = {}
        sens_list = []
        spec_list = []
        j_list = []
        acc_list = []
        for i, (train, test) in enumerate(splits):
            X_train = X[train]
            true_label_train = true_labels[train]
            pmids_train = all_pmids[train]

            X_test = X[test]
            y_test = y[test]
            true_label_test = true_labels[test]
            pmids_test = all_pmids[test]

            pipeline.fit(X_train, true_label_train)
            preds = pipeline.predict(X_test)
            sens = sensitivity_score(y_test, preds, pos_label=-1)
            spec = specificity_score(y_test, preds, pos_label=-1)
            acc = accuracy_score(y_test, preds)
            train_dict = {'pmids': pmids_train.tolist(),
                          'true_label': true_label_train.tolist()}
            test_dict = {'pmids': pmids_test.tolist(), 'true_label':
                         true_label_test.tolist(),
                         'prediction':  preds.tolist(),
                         'true': y_test.tolist()}
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
    main_rng = np.random.RandomState(1729)

    nu_list = [0.1, 0.2, 0.4]
    mf_list = [100, 1000, 10000]
    cases = []
    for model_name in set(available_shortforms.values()):
        for nu in nu_list:
            for mf in mf_list:
                rng = np.random.RandomState(main_rng.randint(2**32))
                cases.append((model_name, nu, mf, rng))

    with Pool(64) as pool:
        pool.map(evaluation_wrapper, cases, chunksize=1)
