import json
import random
import pandas as pd
from multiprocessing import Lock, Pool

from adeft_indra.db.content import get_plaintexts_for_pmids, \
    get_pmids_for_agent_text, get_pmids_for_entity
from adeft_indra.db.anomaly_detection import AnomalyDetectorsManager, \
    ResultsManager
from adeft_indra.ambiguity_detection.find_anomalies import AdeftAnomalyDetector


lock = Lock()
adm = AnomalyDetectorsManager()
results_manager = ResultsManager()


with open('../../results/cord19_ad_blacklist.json') as f:
    blacklist_map = json.load(f)


grounding_table = pd.read_csv('../../data/new_grounding_table.tsv',
                              sep='\t', usecols=['text', 'grounding'])
groundings = [(text, grounding)
              for text, grounding in grounding_table.to_numpy()
              if grounding]
grounding_table = None

param_grid = {'max_features': [100], 'nu': [0.2]}


def anomaly_detect(args):
    agent_text, grounding = args
    ns, id_ = grounding.split(':', maxsplit=1)
    if results_manager.in_table(agent_text, grounding):
        # Results already in table
        with lock:
            print(f'Results for {agent_text}, {grounding} have already'
                  ' been computed.')
        return
    if adm.in_table(grounding):
        # Found pre-existing anomaly detector
        num_grounding_texts, detector = adm.load(grounding)
    else:
        # Gathering content to train new anomaly detector
        pmids = get_pmids_for_entity(ns, id_)
        if not pmids:
            # No pmids found
            return
        if len(pmids) > 10000:
            with lock:
                print(f'Grounding {grounding} has more than 10000'
                      ' associated pmids. Sampling 10000 at random')
            pmids = random.choices(pmids, k=10000)
        grounding_texts = list(get_plaintexts_for_pmids(pmids).values())
        num_grounding_texts = len(grounding_texts)
        blacklist = blacklist_map[grounding]
        detector = AdeftAnomalyDetector(blacklist=blacklist)
        if num_grounding_texts >= 5:
            detector.cv(grounding_texts, [], param_grid, n_jobs=1, cv=5)
            adm.save(grounding, len(grounding_texts), detector)
            with lock:
                print('Anomaly detector trained for grounding'
                      f' {grounding}.')
        else:
            with lock:
                print('Fewer than 5 texts available for grounding'
                      f' {grounding}. Cannot train anomaly detector.')
            # Insufficient data available
            outrow = [agent_text, grounding, num_grounding_texts,
                      None, None, None, None]
            results_manager.add_row(outrow)
    agent_text_pmids = get_pmids_for_agent_text(agent_text)
    if not agent_text_pmids:
        return
    if len(agent_text_pmids) > 10000:
        with lock:
            print(f'Agent text {agent_text} has more than 10000'
                  ' associated pmids. Truncating at random.')
        agent_text_pmids = random.choices(agent_text_pmids,
                                          k=10000)
    pred_content = list(get_plaintexts_for_pmids(agent_text_pmids).values())
    outrow = [agent_text, grounding, num_grounding_texts, len(pred_content)]
    if num_grounding_texts < 5 or len(pred_content) < 1:
        with lock:
            print('Insufficient data available to anomaly detect'
                  f' grounding {grounding} for agent text'
                  f' {agent_text}.')
        outrow += [None, None, None]
        results_manager.add_row(outrow)
    preds = detector.predict(pred_content)
    total_anomalous = sum(preds)
    spec = detector.specificity
    std_spec = detector.std_specificity
    outrow += [total_anomalous, spec, std_spec]
    results_manager.add_row(outrow)
    with lock:
        print(f'Anomaly detection successful for grounding {grounding}'
              f' and agent text {agent_text}.\n'
              f'{",".join(outrow)}')


if __name__ == '__main__':
    n_jobs = 8
    with Pool(n_jobs) as pool:
        pool.map(anomaly_detect, groundings, chunksize=1)
