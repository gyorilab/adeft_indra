import os
import csv
import pickle
from copy import deepcopy
from collections import defaultdict

from gilda.resources import GROUNDING_TERMS_PATH
from adeft.util import SearchTrie, get_candidate

from adeft_indra.locations import GROUNDER_PATH
from adeft_indra.ground.util import expand_dashes, greek_aware_stem, \
    normalize, text_similarity


class AdeftGrounder(object):
    def __init__(self, rebuild=False):
        if rebuild or not os.path.exists(GROUNDER_PATH):
            self._build()
        else:
            with open(GROUNDER_PATH, 'rb') as f:
                self.__dict__.update(pickle.load(f).__dict__)

    def ground(self, text):
        results = []
        expansions = expand_dashes(text)
        for expansion in expansions:
            tokens, longform_map = get_candidate(expansion)
            processed_tokens = [greek_aware_stem(token) for token in tokens]
            match, match_text = self._trie.search(processed_tokens)
            if match is None:
                continue
            entity_tokens, _ = get_candidate(match_text)
            if entity_tokens == processed_tokens[-len(entity_tokens):]:
                longform_text = longform_map[len(entity_tokens)]
                grounding_keys = self.text2index[normalize(longform_text)]
                for grounding_key in grounding_keys:
                    entry = deepcopy(self.index2grounding[grounding_key])
                    entry['longform_text'] = longform_text
                    if len(entity_tokens) == len(processed_tokens):
                        entry['partial_match'] = False
                    else:
                        entry['partial_match'] = True
                    results.append(entry)
        result_dict = {}
        for result in results:
            raw_text = result['raw_text']
            grounding = result['grounding']
            longform_text = result['longform_text']
            score = (text_similarity(longform_text, raw_text),
                     3 - self.type_priority[result['type']])
            if grounding not in result_dict or \
               score > result_dict[grounding]['score']:
                result_dict[grounding] = result
                result_dict[grounding]['score'] = score
        out = [result for result in result_dict.values()
               if result['score'][0] > 0]
        return sorted(out, key=lambda x: (-x['score'][0], -x['score'][1]))

    def _build(self):
        index2grounding, text2index, lx = self._load_default_index2grounding()
        self.index2grounding = index2grounding
        self.text2index = text2index
        self._trie = SearchTrie(lx,
                                expander=expand_dashes,
                                token_map=greek_aware_stem)
        self.type_priority = {'assertion': 0,
                              'name': 1,
                              'synonym': 2,
                              'previous': 3}
        with open(GROUNDER_PATH, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_default_index2grounding(self):
        index2grounding = {}
        text2index = defaultdict(list)
        lexicon = []
        with open(GROUNDING_TERMS_PATH) as f:
            reader = csv.reader(f, delimiter='\t')
            for index, row in enumerate(reader):
                entry = {'grounding': f'{row[2]}:{row[3]}',
                         'type': row[5],
                         'raw_text': row[1],
                         'name': row[4]}
                index2grounding[index] = entry
                text2index[normalize(row[1])].append(index)
                lexicon.append(row[1])
        return index2grounding, text2index, lexicon
