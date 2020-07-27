import csv
from copy import deepcopy
from collections import defaultdict

from gilda.resources import GROUNDING_TERMS_PATH

from adeft.util import SearchTrie, get_candidate
from adeft_indra.ground.util import expand_dashes, greek_aware_stem, \
    normalize, text_similarity


def load_default_index2grounding():
    index2grounding = {}
    text2index = defaultdict(list)
    lexicon = []
    with open(GROUNDING_TERMS_PATH) as f:
        reader = csv.reader(f, delimiter='\t')
        for index, row in enumerate(reader):
            entry = {'grounding': f'{row[2]}:{row[3]}',
                     'type': row[5],
                     'raw_text': row[1]}
            index2grounding[index] = entry
            text2index[normalize(row[1])].append(index)
            lexicon.append(row[1])
    return index2grounding, text2index, lexicon


class AdeftGrounder(object):
    def __init__(self, groundings=None):
        if groundings is None:
            index2grounding, text2index, lx = load_default_index2grounding()
        self.index2grounding = index2grounding
        self.text2index = text2index
        self._trie = SearchTrie(lx,
                                expander=expand_dashes,
                                token_map=greek_aware_stem)
        self.type_priority = {'assertion': 0,
                              'name': 1,
                              'synonym': 2,
                              'previous': 3}

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
