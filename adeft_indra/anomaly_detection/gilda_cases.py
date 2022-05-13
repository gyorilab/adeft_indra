import argparse
from itertools import chain
import logging
import pickle
from multiprocessing import Pool

from collections import defaultdict

from gilda.grounder import load_gilda_models

from indra_db_lite import get_plaintexts_for_text_ref_ids
from indra_db_lite import get_text_ref_ids_for_agent_text

from .cases import get_training_cases_for_grounding


logger = logging.getLogger(__file__)


def get_test_cases_for_model(model):
    assert len(model.shortforms) == 1
    agent_text = model.shortforms[0]
    test_trids = get_text_ref_ids_for_agent_text(agent_text)
    test_texts = get_plaintexts_for_text_ref_ids(test_trids)
    if not test_texts:
        return []
    preds = model.predict(test_texts)
    text_dict = defaultdict(list)
    for text, pred in zip(test_texts, preds):
        text_dict[pred].append(text)
    result = []
    for curie in model.estimator.classes_:
        namespace, identifier = curie.split(":", maxsplit=1)
        train_info = get_training_cases_for_grounding(namespace, identifier)
        if train_info is None:
            continue
        result.append(
            (
                agent_text,
                [agent_text],
                curie,
                train_info["mesh_terms"],
                train_info["num_entrez"],
                train_info["num_mesh"],
                train_info["train_trids"],
                text_dict[curie],
            )
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('outpath')
    parser.add_argument('--n_jobs', type=int, default=1)
    args = parser.parse_args()
    outpath = args.outpath
    models = load_gilda_models()
    n_jobs = args.n_jobs
    with Pool(n_jobs) as pool:
        result = pool.map(get_test_cases_for_model, models.values())
    result = list(chain(*result))
    with open(outpath, 'wb') as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
