from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim.matutils import corpus2csc

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

from adeft_indra.locations import DOCUMENT_FREQUENCIES_PATH


class AdeftTfidfVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, dict_path=None, max_features=None, stop_words=None):
        if dict_path is None:
            dict_path = DOCUMENT_FREQUENCIES_PATH
        self.dict_path = dict_path
        self.max_features = max_features
        self.tokenize = TfidfVectorizer().build_tokenizer()
        if stop_words is None:
            self.stop_words = []
        else:
            self.stop_words = stop_words
        self.model = None
        self.dictionary = None

    def fit(self, raw_documents, y=None):
        # Load background dictionary trained on large corpus
        dictionary = Dictionary.load(self.dict_path)
        # Tokenize training texts and convert tokens to lower case
        processed_texts = (self._preprocess(text) for text in raw_documents)
        local_dictionary = Dictionary(processed_texts)
        local_dictionary.filter_tokens(good_ids=(key for key, value
                                                 in local_dictionary.items()
                                                 if value
                                                 in dictionary.token2id))
        # Remove stopwords
        if self.stop_words:
            stop_ids = [id_ for token, id_ in local_dictionary.token2id.items()
                        if token in self.stop_words]
            local_dictionary.filter_tokens(bad_ids=stop_ids)
        # Keep only most frequent features
        if self.max_features is not None:
            local_dictionary.filter_extremes(no_below=1, no_above=1.0,
                                             keep_n=self.max_features)
        # Filter background dictionary to top features found in
        # training dictionary
        dictionary.filter_tokens(good_ids=(key for key, value
                                           in dictionary.items()
                                           if value
                                           in local_dictionary.token2id))
        model = TfidfModel(dictionary=dictionary)
        self.model = model
        self.dictionary = dictionary
        return self

    def transform(self, raw_documents):
        processed_texts = [self._preprocess(text) for text in raw_documents]
        corpus = (self.dictionary.doc2bow(text) for text in processed_texts)
        transformed_corpus = self.model[corpus]
        X = corpus2csc(transformed_corpus, num_terms=len(self.dictionary))
        return X.transpose()

    def get_feature_names(self):
        return [self.dictionary.id2token[i]
                for i in range(len(self.dictionary))]

    def _preprocess(self, text):
        return [token.lower() for token in self.tokenize(text)]
