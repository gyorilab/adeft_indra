import random
from multiprocessing import Lock
from multiprocessing import Pool
from collections import defaultdict
from sklearn.svm import OneClassSVM
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold


from adeft import available_shortforms
from adeft.nlp import english_stopwords
from adeft.modeling.label import AdeftLabeler
from adeft.disambiguate import load_disambiguator


from adeft_indra.tfidf import AdeftTfidfVectorizer
from adeft_indra.content import get_pmids_for_entity
from adeft_indra.content import get_plaintexts_for_pmids
from adeft_indra.content import get_pmids_for_agent_text
from adeft_indra.model_building.escape import escape_filename
from adeft_indra.db.anomaly_detection import ADResultsManager
from adeft_indra.anomaly_detection.stats import specificity_score
from adeft_indra.anomaly_detection.models import ForestOneClassSVM


lock = Lock()
ADResultsManager('ad_all_adeft')


def get_training_data_for_model(disambiguator):
    ad = disambiguator
    all_pmids = set()
    for shortform in ad.shortforms:
        all_pmids.update(get_pmids_for_agent_text(shortform))
    text_dict = get_plaintexts_for_pmids(all_pmids,
                                         contains=ad.shortforms)

    labeler = AdeftLabeler(ad.grounding_dict)
    corpus = labeler.build_from_texts((text, pmid) for
                                      pmid, text in text_dict.items())
    additional_entities = {}
    unambiguous_agent_texts = {}
    metadata = ad.classifier.other_metadata
    if isinstance(metadata, dict) and 'additional_entities' in metadata:
        additional_entities = metadata['additional_entities']
    if isinstance(metadata, dict) and 'unambiguous_agent_texts' in metadata:
        unambiguous_agent_texts = metadata['unambiguous_agent_texts']
        agent_text_pmid_map = defaultdict(list)
    for text, label, id_ in corpus:
        agent_text_pmid_map[label].append(id_)

    entity_pmid_map = {entity:
                       set(get_pmids_for_entity(*entity.split(':', maxsplit=1),
                                                major_topic=True))
                       for entity in additional_entities}
    all_used_pmids = set()
    for entity, agent_texts in unambiguous_agent_texts.items():
        used_pmids = set()
        for agent_text in agent_texts[1]:
            pmids = set(get_pmids_for_agent_text(agent_text))
            new_pmids = list(pmids - text_dict.keys() - used_pmids)
            _text_dict = get_plaintexts_for_pmids(new_pmids,
                                                  contains=agent_texts)
            corpus.extend([(text, entity, pmid)
                           for pmid, text in _text_dict.items()
                           if len(text) >= 5])
            used_pmids.update(new_pmids)
        all_used_pmids.update(used_pmids)

    for entity, pmids in entity_pmid_map.items():
        new_pmids = list(set(pmids) - text_dict.keys() - all_used_pmids)
        if len(new_pmids) > 10000:
            new_pmids = random.choices(new_pmids, k=10000)
        _, contains = additional_entities[entity]
        _text_dict = get_plaintexts_for_pmids(new_pmids, contains=contains)
        corpus.extend([(text, entity, pmid)
                       for pmid, text in _text_dict.items() if len(text) >= 5])
    all_used_pmids = set(pmid for _, _, pmid in corpus)
    unlabeled = [(text, pmid) for pmid, text in text_dict.items()
                 if pmid not in all_used_pmids and text]
    return corpus, unlabeled


def anomaly_detect_adeft_model(shortform, mf_a, mf_b, nu,
                               n_estimators, rng):
    ad = load_disambiguator(shortform)
    corpus, unlabeled = get_training_data_for_model(ad)
    texts, labels, pmids = zip(*corpus)
    stop_words = english_stopwords
    max_features = mf_a + mf_b * len(ad.labels)
    pipeline = Pipeline([('tfidf',
                          AdeftTfidfVectorizer(max_features=max_features,
                                               stop_words=stop_words)),
                         ('forest_oc_svm',
                          ForestOneClassSVM(nu=nu,
                                            cache_size=1000,
                                            n_estimators=n_estimators,
                                            n_jobs=1,
                                            random_state=rng))])
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=rng).\
        split(texts, labels)
    scorer = {'spec': make_scorer(specificity_score, pos_label=-1.0)}
    scores = cross_validate(pipeline, texts, labels,
                            scoring=scorer, cv=folds)
    pipeline.fit(texts, labels)
    out_texts, out_pmids = zip(*unlabeled)
    preds = pipeline.predict(out_texts)
    results = {'pmids': out_pmids, 'preds': preds, 'scores': scores}
    return results