import csv
import json
import pickle
import pandas as pd
from collections import defaultdict
from indra.databases.uniprot_client import get_hgnc_id
from indra.databases.hgnc_client import get_hgnc_name
from indra.databases.hgnc_client import get_hgnc_id as hc_get_hgnc_id
from indra.preassembler.grounding_mapper.mapper import \
    _load_default_grounding_map

from statsmodels.stats.proportion import proportion_confint

from indra_db.util.content_scripts import get_text_content_from_stmt_ids, \
    get_text_content_from_pmids, get_stmts_with_agent_text_like

from adeft import available_shortforms
from adeft.disambiguate import load_disambiguator
from adeft.modeling.find_anomalies import AdeftAnomalyDetector


stmts = get_stmts_with_agent_text_like('c-MET')
stmts.update(get_stmts_with_agent_text_like('c-met'))
stmts.update(get_stmts_with_agent_text_like('c-src'))
stmts.update(get_stmts_with_agent_text_like('Tpl2'))
stmts.update(get_stmts_with_agent_text_like('HK2'))
stmts.update(get_stmts_with_agent_text_like('Fes'))
stmts.update(get_stmts_with_agent_text_like('p56'))



with open('../data/entrez_all_pmids.json', 'r') as f:
    pmids = json.load(f)

with open('../data/hgnc_groundings.pkl', 'rb') as f:
    groundings = pickle.load(f)

kinases = pd.read_csv('../data/allsources_HMS_it2_cleaned.csv', sep=',')

light_kinase_symbols = [get_hgnc_name(get_hgnc_id(entry)) for entry in
                        kinases[kinases.IDG_dark == 0].Entry]
light_kinase_symbols = list(set(light_kinase_symbols))
light_kinase_ids = [hc_get_hgnc_id(symbol) for symbol in light_kinase_symbols]

gm = _load_default_grounding_map()
gene_gm = {key: value for key, value in gm.items() if 'HGNC' in value}
kinase_gm = {key: value for key, value in gene_gm.items() if value['HGNC'] in light_kinase_ids}

gm_kinase_symbols = {key: get_hgnc_name(value['HGNC']) for key, value in kinase_gm.items()}

kinase_scores = pd.read_csv('../results/light_kinase_scores.csv')
processed = kinase_scores['gene_name'].unique()
light_kinase_groundings = {key: value for key, value in groundings.items()
                           if key in light_kinase_symbols and key not in processed}

gm_kinase_groundings = defaultdict(dict)
for text, gene in gm_kinase_symbols.items():
    gm_kinase_groundings[gene].update({text: [(x, '') for x in stmts[text]]})


param_grid = {'max_features': [100],
              'nu': [0.2]}
with open('../results/light_kinase_scores_grounding_mapped.csv', 'w', newline='') as f:
    writer = csv.writer(f, delimiter=',', quotechar='"',
                        quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['gene_name', 'agent_text', 'num_stmts', 'num_entrez',
                     'CI_lower', 'CI_upper', 'adeft_lower', 'adeft_upper',
                     'theta_lower', 'theta_upper'])
    for gene, grounding_info in gm_kinase_groundings.items():
        gene_pmids = pmids.get(gene)
        if gene_pmids:
            _, entrez_content = get_text_content_from_pmids(pmids[gene])
        else:
            continue
        entrez_content = list(entrez_content.values())
        # blacklist all agent texts grounding to this gene with
        # more than 10 associated statement
        blacklist = [key for key, value in grounding_info.items()
                     if len(value) > 10]
        if len(entrez_content) > 4:
            detector = AdeftAnomalyDetector('../results/pubmed_dictionary.pkl',
                                            blacklist=blacklist)
            detector.cv(entrez_content, [], param_grid, n_jobs=8, cv=5)
        for agent_text, stmts in grounding_info.items():
            stmts = [x[0] for x in stmts]
            if stmts:
                _, content = get_text_content_from_stmt_ids(stmts)
            else:
                continue
            if not content:
                continue
            content = list(content.values())
            if agent_text in available_shortforms and len(content) > 1:
                disamb = load_disambiguator(agent_text)
                preds = disamb.disambiguate(content)
                a, b = proportion_confint(len([pred for pred
                                               in preds
                                               if pred[1] != gene]),
                                          len(preds), alpha=0.95,
                                          method='jeffreys')
            else:
                a, b = (0, 1)
            if len(entrez_content) > 4 and len(content) > 1:
                CI = detector.confidence_interval(content,
                                                  sensitivity=0.75)
                preds = detector.predict(content)
                c, d = proportion_confint(sum(preds), len(preds),
                                          alpha=0.95,
                                          method='jeffreys')
            else:
                CI = (0, 1)
                c, d = (0, 1)
            row = [gene, agent_text, len(stmts), len(entrez_content),
                   CI[0], CI[1], a, b, c, d]
            print(row[0:6])
            writer.writerow(row)
