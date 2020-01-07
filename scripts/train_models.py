import os
import pickle

from indra.databases.hgnc_client import get_hgnc_name
from indra.literature.adeft_tools import universal_extract_text

from adeft.disambiguate import load_disambiguator
from adeft.modeling.find_anomalies import AdeftAnomalyDetector

from adeft_indra.s3 import escape_filename

def build_models(shortform):
    disambiguator = load_disambiguator(shortform)
    with open(f'../data/content/{shortform}_content.pkl', 'rb') as f:
        content = pickle.load(f)
    texts = [universal_extract_text(text, contains=[shortform])
             for text in content.values() if text]
    disambs = disambiguator.disambiguate(texts)
    for grounding in disambiguator.names:
        if grounding.startswith('HGNC'):
            hgnc_id = grounding.rsplit(':')[1]
            hgnc_name = get_hgnc_name(hgnc_id)
            try:
                with open(f'../data/entrez_content/{hgnc_name}_content.pkl',
                          'rb') as f:
                    entrez_content = pickle.load(f)
            except Exception:
                print(f'no content for gene {grounding} with shortform'
                      f' {shortform}')
                continue
            entrez_texts = [universal_extract_text(text)
                            for text in entrez_content if text]
            detector = AdeftAnomalyDetector(blacklist=[shortform])
            anomalous_texts = [text for text, disamb in zip(texts, disambs) if
                               disamb[0] != grounding and
                               set(disamb[2].values()) == set([0.0, 1.0])]
            param_grid = {'nu': [0.01, 0.05, 0.1, 0.2],
                          'max_features': [10, 50, 100],
                          'ngram_range': [(1, 1)]}
            try:
                detector.cv(entrez_texts, anomalous_texts, param_grid,
                            n_jobs=10, cv=5)
            except Exception:
                print(f'no samples for gene {grounding} with shortform'
                      f' {shortform}')
                continue
            feature_importance = detector.feature_importances()
            sensitivity = detector.sensitivity
            specificity = detector.specificity
            params = detector.best_params
            cv_results = detector.cv_results
            result = {'fi': feature_importance, 'sens': sensitivity,
                      'spec': specificity, 'params': params, 'cv': cv_results}
            directory = os.path.join('..', 'results',
                                     escape_filename(shortform),
                                     grounding)
            if not os.path.exists(directory):
                os.makedirs(directory)
            with open(os.path.join(directory, 'results.pkl'), 'wb') as f:
                pickle.dump(result, f) 

gene_shortforms = ['AD', 'GH', 'MB', 'PC', 'RB', 'ARF', 'GR', 'MCT', 'PD1',
                   'SP', 'AR', 'GSC', 'MS', 'PE', 'TF', 'CS', 'HK2',
                   'NE', 'PGP', 'TGH', 'ER', 'HR', 'NIS', 'PKD', 'TG',
                   'GC', 'IR', 'NP', 'PS', 'UBC']
if __name__ == '__main__':
    for shortform in gene_shortforms:
        build_models(shortform)
