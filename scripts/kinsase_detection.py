import json
import pickle
import pandas as pd
from collections import defaultdict
from indra.databases.uniprot_client import get_hgnc_id
from indra.databases.hgnc_client import get_hgnc_name

from statsmodels.stats.proportion import proportion_confint

from indra_db.util.content_scripts import get_text_content_from_stmt_ids, \
    get_text_content_from_pmids

from adeft import available_shortforms
from adeft.disambiguate import load_disambiguator
from adeft.modeling.find_anomalies import AdeftAnomalyDetector


with open('../data/entrez_all_pmids.json', 'r') as f:
    pmids = json.load(f)

with open('../data/hgnc_groundings.pkl', 'rb') as f:
    groundings = pickle.load(f)

kinases = pd.read_csv('../data/allsources_HMS_it2_cleaned.csv', sep=',')

light_kinase_symbols = [get_hgnc_name(get_hgnc_id(entry)) for entry in
                        kinases[kinases.IDG_dark == 0].Entry]

light_kinase_groundings = {key: value for key, value in groundings.items()
                           if key in light_kinase_symbols}

param_grid = {'max_features': [100], 'ngram_range': [(1, 1)], 'nu': [0.2]}
with open('../results/light_kinase_scores.tsv', 'w') as f:
    f.write('gene_name\tagent_text\tnum_stmts\tnum_entrez\t'
            'CI_lower\tCI_upper\tadeft_lower\tadeft_upper\t'
            'theta_lower\ttheta_upper\n')
    for gene, grounding_info in light_kinase_groundings.items():
        try:
            _, entrez_content = get_text_content_from_pmids(pmids[gene])
        except KeyError:
            continue
        entrez_content = list(entrez_content.values())
        # blacklist all agent texts grounding to this gene with
        # more than 10 associated statement
        blacklist = [key for key, value in grounding_info.items()
                     if len(value) > 10]
        if len(entrez_content) > 4:
            detector = AdeftAnomalyDetector(blacklist=blacklist)
            detector.cv(entrez_content, [], param_grid, n_jobs=8, cv=5)
        for agent_text, stmts in grounding_info.items():
            stmts = [x[0] for x in stmts]
            _, content = get_text_content_from_stmt_ids(stmts)
            content = list(content.values())
            if agent_text in available_shortforms:
                disamb = load_disambiguator(agent_text)
                preds = disamb.disambiguate(content)
                a, b = proportion_confint(len([pred for pred
                                               in preds
                                               if pred[1] != gene]),
                                          len(preds), alpha=0.95,
                                          method='jeffreys')
            else:
                a, b = (0, 1)

            if len(entrez_content) > 4:
                CI = detector.confidence_interval(content,
                                                  sensitivity=0.75)
                preds = detector.predict(content)
                c, d = proportion_confint(sum(preds), len(preds),
                                          alpha=0.95,
                                          method='jeffreys')
            else:
                CI = (0, 1)
                c, d = (0, 1)
            out = (f'{gene}\t{agent_text}'
                   f'\t{len(stmts)}\t{len(entrez_content)}\t{CI[0]}'
                   f'\t{CI[1]}\t{a}\t{b}\t{c}\t{d}\n')
            print(out)
            f.write(out)
