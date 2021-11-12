import argparse
import json
import random
import subprocess

from gensim.corpora import Dictionary


from indra_db_lite import get_plaintexts_for_text_ref_ids
from indra_db_lite import get_text_ref_ids_for_pmids

from opaque.nlp.featurize import BaselineTfidfVectorizer


class TextFuzzer:
    def __init__(self):
        vectorizer = BaselineTfidfVectorizer()
        dictionary = Dictionary.load(vectorizer.path)
        dictionary.filter_extremes(no_above=0.05, no_below=5)
        self.vectorizer = vectorizer
        self.dictionary = dictionary

    def __call__(self, text):
        tokens = self.vectorizer._preprocess(text)
        tokens = [
            token for token in tokens if token in self.dictionary.token2id
        ]
        if {'xml', 'elsevier', 'doi', 'article'} <= set(tokens):
            return None
        random.shuffle(tokens)
        return ' '.join(tokens)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('inpath')
    parser.add_argument('outpath')
    parser.add_argument('n_threads')
    args = parser.parse_args()
    inpath = args.inpath
    outpath = args.outpath
    n_threads = args.n_threads
    with open(inpath) as f:
        pmids = json.load(f)
    pmids = [int(pmid) for pmid in pmids]
    trids = get_text_ref_ids_for_pmids(pmids).values()
    content = get_plaintexts_for_text_ref_ids(
        trids, text_types=['fulltext']
    )
    fuzz_text = TextFuzzer()
    result = {
        trid: fuzz_text(text)
        for trid, text in content.trid_content_pairs()
    }
    result = {
        trid: text for trid, text in result.items() if text is not None
        and len(text) > 500
    }
    gen = Random(1729)
    sample = set(gen.sample(list(result.keys()), k=50000))
    sample_texts = {
        trid: text for trid: text in result.items()
        if trid in sample
    }
    with open(outpath, 'w') as f:
        json.dump(sample_texts, f, indent=True)
    with open(outpath + '.full', 'w') as f:
        json.dump(result, f, indent=True)
