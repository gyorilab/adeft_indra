import json
import numpy as np
from inspect import getsource
from itertools import product
from multiprocessing import Lock
from multiprocessing import Pool
from collections import defaultdict
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split

from indra.literature.adeft_tools import filter_paragraphs

from adeft import available_shortforms
from adeft.modeling.label import AdeftLabeler
from adeft.disambiguate import load_disambiguator

from opaque.ood.svm import OneClassLinearSVC
from opaque.ood.svm import SerializableOneClassSVM
from opaque.nlp.models import GroundingAnomalyDetector
from opaque.nlp.featurize import BaselineTfidfVectorizer
from opaque.stats import sensitivity_score, specificity_score


from adeft_indra.content import get_pmids_for_agent_text
from adeft_indra.content import get_paragraphs_for_pmids
from adeft_indra.locations import DOCUMENT_FREQUENCIES_PATH
from adeft_indra.model_building.escape import escape_filename
from adeft_indra.db.anomaly_detection import ADResultsManager


lock = Lock()
manager = ADResultsManager('linear_svc_tune1')
reverse_model_map = {value: key for
                     key, value in available_shortforms.items()}


def evaluation_wrapper(args):
    model_name, nu, max_features, smartirs, no_above, no_below, rng = args
    if callable(max_features):
        mf_param = getsource(max_features)
    else:
        mf_param = max_features
    params = {
        'nu': nu,
        'max_features': mf_param,
        'smartirs': smartirs,
        'no_above': no_above,
        'no_below': no_below
    }
    if manager.in_table(model_name, json.dumps(params)):
        with lock:
            print(f'Results for {model_name}, {params} have already'
                  ' been computed')
        return
    model = load_disambiguator(reverse_model_map[model_name])
    results = evaluate_anomaly_detection(
        model,
        nu,
        max_features,
        smartirs,
        no_above,
        no_below,
        rng
    )

    if results:
        manager.add_row([model_name, json.dumps(params), json.dumps(results)])
        with lock:
            print(f'Success for {model_name}, params={params}')
    else:
        with lock:
            print(f'Problem with {model_name}, params={params}')


def evaluate_anomaly_detection(
        model,
        nu,
        max_features,
        smartirs,
        no_above,
        no_below,
        rng
):
    ad = model
    model_name = escape_filename('&'.join(sorted(ad.shortforms)))
    if callable(max_features):
        mf_param = getsource(max_features)
    else:
        mf_param = max_features

    params = {'nu': nu, "max_features": mf_param,
              "no_above": no_above, "no_below": no_below,
              "smartirs": smartirs}
    with lock:
        print(f'Working on {model_name} with params {params}')
    stop_words = set(ad.shortforms)
    all_pmids = set()
    for shortform in ad.shortforms:
        all_pmids.update(get_pmids_for_agent_text(shortform))
    paragraphs_dict = get_paragraphs_for_pmids(all_pmids)

    labeler = AdeftLabeler(ad.grounding_dict)
    corpus = labeler.build_from_texts(
        (filter_paragraphs(pars), pmid) for pmid, pars
        in paragraphs_dict.items()
    )

    entity_pmid_map = defaultdict(list)
    for _, label, pmid in corpus:
        entity_pmid_map[label].append(pmid)
    entity_pmid_map = dict(entity_pmid_map)

    grounded_entities = [(entity, len(entity_pmid_map[entity]))
                         for entity in entity_pmid_map
                         if entity != 'ungrounded']

    grounded_entities = [(entity, num_pmids) for entity, num_pmids in
                         grounded_entities if num_pmids >= 20]

    grounded_entities.sort(key=lambda x: -x[1])
    results = defaultdict(dict)
    for entity, _ in grounded_entities:
        inlier_pmids = {
            pmid
            for label, pmids in entity_pmid_map.items()
            for pmid in pmids if pmid in entity_pmid_map[entity]
        }
        inlier = [
            (
                filter_paragraphs(paragraphs_dict[pmid]),
                filter_paragraphs(
                    paragraphs_dict[pmid], contains=ad.shortforms
                ),
                pmid
            )
            for pmid in inlier_pmids
        ]

        if not inlier:
            continue

        texts, texts_filtered, pmids = zip(*inlier)
        texts, texts_filtered, pmids = (
            list(texts), list(texts_filtered), list(pmids)
        )
        inlier = None

        anomalous_pmids = {
            pmid
            for label, pmids in entity_pmid_map.items()
            for pmid in pmids if pmid not in entity_pmid_map[entity]
        }

        anomalous = (
            (
                filter_paragraphs(
                    paragraphs_dict[pmid], contains=ad.shortforms
                ),
                pmid
            )
            for pmid in anomalous_pmids
        )

        try:
            anomalous_texts, anomalous_ = zip(*anomalous)
        except ValueError:
            print
            continue
        anomalous_texts = list(anomalous_texts)
        anomalous_pmids = list(anomalous_pmids)

        rng2 = np.random.RandomState(561)
        rng2.shuffle(texts)
        rng2 = np.random.RandomState(561)
        rng2.shuffle(pmids)
        rng2 = None

        train_splits = KFold(
            n_splits=5,
            shuffle=False
        ).split(texts)

        Xin = np.array(texts)
        Xin_filtered = np.array(texts_filtered)
        Xout = np.array(anomalous_texts)
        texts = None
        texts_filtered = None
        anomalous_texts = None
        pmids_in = np.array(pmids)
        pmids_out = np.array(anomalous_pmids)
        pmids = None
        anomalous_pmids = None
        folds_dict = {}
        sens_list = []
        spec_list = []
        j_list = []
        acc_list = []
        for i, (train, test) in enumerate(train_splits):
            ad_model = GroundingAnomalyDetector(


                /*BaselineTfidfVectorizer(
                    DOCUMENT_FREQUENCIES_PATH,
                    max_features_per_class=max_features,
                    no_above=no_above,
                    no_below=no_below,
                    smartirs=smartirs,
                    stop_words=stop_words
                ),
                SerializableOneClassSVM(kernel="linear", nu=nu)
            )
            ad_model.fit(Xin[train])
            preds_in = ad_model.predict(Xin_filtered[test[0]:test[-1]+1])
            preds_out = ad_model.predict(Xout)
            ad_model = None
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
            train_dict = {'pmids': pmids_in[train].tolist()}
            test_dict = {'pmids': pmids_in[test[0]:test[-1]+1].tolist() +
                         pmids_out.tolist(),
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
        results[entity] = {
            'folds': folds_dict,
            'results': aggregate_scores,
            'params': params
        }
    return results


if __name__ == '__main__':
    model_data_counts = []
    for model_name in set(available_shortforms.values()):
        ad = load_disambiguator(reverse_model_map[model_name])
        stats = ad.classifier.stats
        total_count = sum(stats['label_distribution'].values())
        model_data_counts.append((model_name, total_count))
    model_data_counts.sort(key=lambda x: -x[1])

    # Don't use high data models, it will take too long
    models_used = [x[0] for x in model_data_counts[10:]]

    main_rng = np.random.RandomState(1729)
    train, validation = train_test_split(models_used, random_state=main_rng)

    print('validation models', validation)

    rows = manager.get_results()
    nu_list = [0.15]
    max_features_list = [15]
    smartirs_list = ['ntc']
    no_above_list = [0.05]
    no_below_list = [5]
    cases = []
    feature_combs = product(
        nu_list,
        max_features_list,
        smartirs_list,
        no_above_list,
        no_below_list
    )
    for nu, max_features, smartirs, no_above, no_below in feature_combs:
        for model_name in validation:
            rng = np.random.RandomState(main_rng.randint(2**32))
            cases.append(
                (
                    model_name,
                    nu,
                    max_features,
                    smartirs,
                    no_above,
                    no_below,
                    rng
                )
            )
    main_rng.shuffle(cases)

    with Pool(64) as pool:
        pool.map(evaluation_wrapper, cases, chunksize=1)
