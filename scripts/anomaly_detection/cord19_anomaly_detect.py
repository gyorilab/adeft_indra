import csv
import json
import random

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

param_grid = {'max_features': [100], 'nu': [0.2]}
with open('../../data/new_grounding_table.tsv', newline='') as csvfile:
    rows = csv.DictReader(csvfile, delimiter='\t')
    for i, row in enumerate(rows):
        grounding = row['grounding']
        if not grounding:
            continue
        ns, id_ = grounding.split(':', maxsplit=1)
        agent_text = row['text']
        print(f'{i}: Anomaly detector for {agent_text}, {grounding}')
        if results_manager.in_table(agent_text, grounding):
            print('Results already exist in table')
            continue
        if adm.in_table(grounding):
            print('Found pre-existing AnomalyDetector')
            num_grounding_texts, detector = adm.load(grounding)
        else:
            print('Gathering content to train new Anomaly Detector')
            pmids = pmid_map.get(grounding)
            if not pmids:
                continue
            if len(pmids) > 10000:
                pmids = random.choices(pmids, k=10000)
                grounding_texts = list(get_plaintexts_for_pmids(pmids).
                                       values())
            num_grounding_texts = len(grounding_texts)
            blacklist = blacklist_map[grounding]
            detector = AdeftAnomalyDetector(blacklist=blacklist)
            if num_grounding_texts >= 5:
                detector.cv(grounding_texts, [], param_grid, n_jobs=5, cv=5)
                adm.save(grounding, len(grounding_texts), detector)
            else:
                print('Insufficient data available')
                outrow = [agent_text, grounding, num_grounding_texts,
                          None, None, None, None]
                print(outrow)
                results_manager.add_row(outrow)
                continue
        print(f'Gathering content for entity text {agent_text}')
        agent_text_pmids = get_pmids_for_agent_text(agent_text)
        if not agent_text_pmids:
            continue
        if len(agent_text_pmids) > 10000:
            agent_text_pmids = random.choices(agent_text_pmids,
                                              k=10000)
        pred_content = list(get_plaintexts_for_pmids(agent_text_pmids).
                            values())
        outrow = [agent_text, grounding, num_grounding_texts,
                  len(pred_content)]
        if num_grounding_texts < 5 or len(pred_content) < 1:
            print('Insufficient data available')
            outrow += [None, None, None]
            print(outrow)
            results_manager.add_row(outrow)
        print('Making predictions with anomaly detector')
        preds = detector.predict(pred_content)
        total_anomalous = sum(preds)
        spec = detector.specificity
        std_spec = detector.std_specificity
        print('Results')
        outrow += [total_anomalous, spec, std_spec]
        print(outrow)
        results_manager.add_row(outrow)
