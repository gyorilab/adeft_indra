import csv
import json
from indra_db.util.content_scripts import get_stmts_with_agent_text_like

from adeft_indra.db.content import ContentCache
from adeft_indra.db.anomaly_detection import AnomalyDetectorsManager, \
    ResultsManager
from adeft_indra.ambiguity_detection.find_anomalies import AdeftAnomalyDetector

adm = AnomalyDetectorsManager()
cc = ContentCache()
results_manager = ResultsManager()


with open('../../results/cord19_entity_pmids2.json') as f:
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
        if ns not in ['CHEBI', 'FPLX', 'HGNC']:
            continue
        agent_text = row['text']
        print(f'{i}: Anomaly detector for {agent_text}, {grounding}')
        if results_manager.in_table(agent_text, grounding):
            print('Results already exist in table')
            continue
        if adm.in_table(grounding):
            print('Found pre-existing AnomalyDetector')
            detector = adm.load(grounding)
        else:
            print('Gathering content to train new Anomaly Detector')
            pmids = pmid_map.get(grounding)
            if not pmids:
                continue
            grounding_texts = cc.get_text_content_from_pmids(pmids, njobs=8)
            blacklist = blacklist_map[grounding]
            detector = AdeftAnomalyDetector(blacklist=blacklist)
            if len(grounding_texts) >= 5:
                detector.cv(grounding_texts, [], param_grid, n_jobs=5, cv=5)
                adm.save(grounding, detector)
            else:
                print('Insufficient data available')
                outrow = [agent_text, grounding, len(grounding_texts),
                          None, None, None, None]
                print(outrow)
                results_manager.add_row(outrow)
                continue
        print(f'Gathering content for entity text {agent_text}')
        agent_stmts = \
            get_stmts_with_agent_text_like(agent_text)[agent_text]
        if not agent_stmts:
            continue
        agent_texts = cc.get_text_content_from_stmt_ids(agent_stmts,
                                                        njobs=5)
        outrow = [agent_text, grounding, len(grounding_texts),
                  len(agent_texts)]
        if len(grounding_texts) < 5 or len(agent_texts) < 1:
            print('Insufficient data available')
            outrow += [None, None, None]
            print(outrow)
            results_manager.add_row(outrow)
        print('Making predictions with anomaly detector')
        preds = detector.predict(agent_texts)
        total_anomalous = sum(preds)
        spec = detector.specificity
        std_spec = detector.std_specificity
        print('Results')
        outrow += [total_anomalous, spec, std_spec]
        print(outrow)
        results_manager.add_row(outrow)
