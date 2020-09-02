import csv
import json
import random
import pandas as pd
from multiprocessing import Pool


from adeft_indra.db.content import get_plaintexts_for_pmids, \
    get_pmids_for_agent_text
from adeft_indra.db.anomaly_detection import AnomalyDetectorsManager, \
    ResultsManager
from adeft_indra.ambiguity_detection.find_anomalies import AdeftAnomalyDetector


adm = AnomalyDetectorsManager()
results_manager = ResultsManager()


with open('../../results/cord19_entity_pmids3.json') as f:
    pmid_map = json.load(f)

with open('../../results/cord19_ad_blacklist.json') as f:
    blacklist_map = json.load(f)


grounding_table = pd.read_csv('../../data/new_grounding_table.tsv',
                              sep='\t', usecols=['text', 'grounding'])
groundings = [(text, grounding)
              for text, grounding in grounding_table.to_list()
              if grounding]

param_grid = {'max_features': [100], 'nu': [0.2]}


def anomaly_detect(agent_text, grounding):
    ns, id_ = grounding.split(':', maxsplit=1)
    if results_manager.in_table(agent_text, grounding):
        # Results already in table
        return
    if adm.in_table(grounding):
        # Found pre-existing anomaly detector
        num_grounding_texts, detector = adm.load(grounding)
    else:
        # Gathering content to train new anomaly detector
        pmids = pmid_map.get(grounding)
        if not pmids:
            # No pmids found
            return
        if len(pmids) > 10000:
            pmids = random.choices(pmids, k=10000)
            grounding_texts = list(get_plaintexts_for_pmids(pmids).values())
        num_grounding_texts = len(grounding_texts)
        blacklist = blacklist_map[grounding]
        detector = AdeftAnomalyDetector(blacklist=blacklist)
        if num_grounding_texts >= 5:
            detector.cv(grounding_texts, [], param_grid, n_jobs=5, cv=5)
            adm.save(grounding, len(grounding_texts), detector)
        else:
            # Insufficient data available
            outrow = [agent_text, grounding, num_grounding_texts,
                      None, None, None, None]
            results_manager.add_row(outrow)
    agent_text_pmids = get_pmids_for_agent_text(agent_text)
    if not agent_text_pmids:
        return
    if len(agent_text_pmids) > 10000:
        agent_text_pmids = random.choices(agent_text_pmids,
                                          k=10000)
    pred_content = list(get_plaintexts_for_pmids(agent_text_pmids).values())
    outrow = [agent_text, grounding, num_grounding_texts, len(pred_content)]
    if num_grounding_texts < 5 or len(pred_content) < 1:
        outrow += [None, None, None]
        results_manager.add_row(outrow)
    preds = detector.predict(pred_content)
    total_anomalous = sum(preds)
    spec = detector.specificity
    std_spec = detector.std_specificity
    outrow += [total_anomalous, spec, std_spec]
    results_manager.add_row(outrow)


if __name__ == '__main__':
    n_jobs = 16
    with Pool(n_jobs) as pool:
        pool.map(anomaly_detect, groundings)
