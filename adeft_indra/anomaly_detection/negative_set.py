import argparse
import json
import random
import subprocess

from gensim.corpora import Dictionary

from indra_db_lite import get_text_sample

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
    parser.add_argument('outpath')
    parser.add_argument('n_threads')
    args = parser.parse_args()
    outpath = args.outpath
    n_threads = args.n_threads
    content = get_text_sample(100000, text_types=['fulltext'])
    content.process()
    fuzz_text = TextFuzzer()
    result = {
        trid: fuzz_text(text)
        for trid, text in content.trid_content_pairs()
    }
    result = {
        trid: text for trid, text in result.items() if text is not None
        and len(text) > 500
    }
    with open(outpath, 'w') as f:
        json.dump(result, f, indent=True)
    subprocess.run(
        ["xz", "-v", "-1", n_threads, outpath]
    )
