import logging

from adeft import available_shortforms
from adeft.disambiguate import load_disambiguator
from adeft.modeling.label import AdeftLabeler

from indra.databases.hgnc_client import get_uniprot_id

from indra_db_lite import get_entrez_pmids_for_hgnc
from indra_db_lite import get_entrez_pmids_for_uniprot
from indra_db_lite import get_mesh_terms_for_grounding
from indra_db_lite import get_plaintexts_for_text_ref_ids
from indra_db_lite import get_pmids_for_mesh_term
from indra_db_lite import get_text_ref_ids_for_agent_text
from indra_db_lite import get_text_ref_ids_for_pmids

__all__ = ["get_adeft_test_cases"]


logger = logging.getLogger(__file__)


models = {
    model_name: load_disambiguator(shortform)
    for shortform, model_name in available_shortforms.items()
}

reverse_model_map = {
    model_name: shortform for shortform, model_name
    in available_shortforms.items()
}


def get_groundings_for_disambiguator(disamb):
    result = set()
    for grounding_map in disamb.grounding_dict.values():
        for curie in grounding_map.values():
            result.add(curie)
    return list(result)


def get_test_cases_for_model(model_name):
    disamb = models[model_name]
    shortforms = disamb.shortforms
    trids = set()
    for shortform in shortforms:
        trids.update(get_text_ref_ids_for_agent_text(shortform))
    trids = list(trids)
    content = get_plaintexts_for_text_ref_ids(
        trids, contains=shortforms, text_types=['abstract', 'fulltext']
    )
    unlabeled = [
        (trid, text) for trid, text in content.trid_content_pairs()
        if len(text) > 5
    ]
    labeler = AdeftLabeler(grounding_dict=disamb.grounding_dict)
    test_corpus = labeler.build_from_texts(
        (text, trid) for trid, text in unlabeled
    )
    test_texts, test_labels, test_trids = zip(*test_corpus)
    test_corpus = None
    for curie in get_groundings_for_disambiguator(disamb):
        if ':' not in curie:
            continue
        if len([label for label in test_labels if label == curie]) < 5:
            continue
        namespace, identifier = curie.split(':', maxsplit=1)
        mesh_id = None
        entrez_pmids = []
        mesh_pmids = []
        if namespace == 'HGNC':
            entrez_pmids = get_entrez_pmids_for_hgnc(identifier)
            uniprot_id = get_uniprot_id(identifier)
            mesh_terms = get_mesh_terms_for_grounding(namespace, identifier)
            if not mesh_terms:
                mesh_terms = get_mesh_terms_for_grounding('UP', uniprot_id)
            if mesh_terms:
                mesh_id = mesh_terms[0]
                mesh_pmids = get_pmids_for_mesh_term(
                    mesh_id, major_topic=True
                )
        elif namespace == 'UP':
            entrez_pmids = get_entrez_pmids_for_uniprot(identifier)
            mesh_terms = get_mesh_terms_for_grounding(namespace, identifier)
            if mesh_terms:
                mesh_id = mesh_terms[0]
                mesh_pmids = get_pmids_for_mesh_term(
                    mesh_id, major_topic=True
                )
        elif namespace == 'MESH':
            mesh_id = identifier
            mesh_pmids = get_pmids_for_mesh_term(
                mesh_id, major_topic=True
            )
        else:
            mesh_terms = get_mesh_terms_for_grounding(namespace, identifier)
            if mesh_terms:
                mesh_id = mesh_terms[0]
                mesh_pmids = get_pmids_for_mesh_term(
                    mesh_id, major_topic=True
                )
        if entrez_pmids:
            entrez_trids = list(
                get_text_ref_ids_for_pmids(entrez_pmids).values()
            )
            train_data = get_plaintexts_for_text_ref_ids(
                entrez_trids, text_types=['fulltext', 'abstract']
            )
            train_data = [
                (trid, text) for trid, text in train_data.trid_content_pairs()
                if len(text) > 5
            ]
            if len(train_data) < 5:
                continue
            train_trids, train_texts = zip(*train_data)
            train_data = None
            yield (
                model_name,
                tuple(shortforms),
                curie,
                'entrez',
                None,
                train_trids,
                train_texts,
                test_trids,
                test_labels,
                test_texts,
            )
        if mesh_pmids:
            mesh_trids = list(get_text_ref_ids_for_pmids(mesh_pmids).values())
            train_data = get_plaintexts_for_text_ref_ids(
                mesh_trids, text_types=['fulltext', 'abstract']
            )
            train_data = [
                (trid, text) for trid, text in train_data.trid_content_pairs()
                if len(text) > 5
            ]
            if len(train_data) < 5:
                continue
            train_trids, train_texts = zip(*train_data)
            train_data = None
            yield(
                model_name,
                tuple(shortforms),
                curie,
                'mesh',
                mesh_id,
                train_trids,
                train_texts,
                test_trids,
                test_labels,
                test_texts,
            )


def get_adeft_test_cases():
    for model_name in models:
        for row in get_test_cases_for_model(model_name):
            yield row
