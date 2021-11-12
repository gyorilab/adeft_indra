import re
from fuzzywuzzy import fuzz
from itertools import chain, combinations, product

from adeft.nlp import stem
from adeft.util import get_candidate


def _equivalence_helper(text1, text2):
    if text1.lower() == text2.lower():
        return True
    stem1, stem2 = stem(text1), stem(text2)
    if text1.endswith('ies') and stem1.endswith('i') \
       and text2.endswith('y') and text2[:-1].lower() == stem2[:-1]:
        return True
    if text1.endswith('es') and text1[:-2].lower() == text2.lower():
        return True
    if text1.endswith('es') and text2.endswith('e') and \
       text1[:-2].lower() == text2[:-1].lower():
        return True
    if text1.endswith('s') and text1[:-1].lower() == text2.lower():
        return True
    return False


def equivalent_up_to_plural(text1, text2):
    """Return True if one of text1, text2 is a plural form of the other.

    Plurality is detected naively. This approach is flawed but appears to work
    well enough.

    Parameters
    ----------
    text1 : str
    text2 : str

    Returns
    -------
    bool
    """
    return (_equivalence_helper(text1, text2) or
            _equivalence_helper(text2, text1))


def text_similarity(text, grounding_text):
    """Compute a similarity score between a text and a known grounded text

    Parameters
    ----------
    text : str
        Some text seen in the wild.
    grounding_text : str
        A text from a known dictionary of grounded texts.

    Returns
    -------
    float
        Fuzzy matching for grounding texts with multiple tokens, strict matching
        for one token texts.
    """
    if text.lower() == grounding_text.lower():
        output = 1.0
    elif len(get_candidate(grounding_text)[0]) > 1:
        output = fuzz.ratio(text.lower(), grounding_text.lower())/100
    elif equivalent_up_to_plural(text, grounding_text):
        output = 0.95
    else:
        output = 0.0
    return output


def greek_aware_stem(text):
    """Stemmer that maps greek unicode letters to ascii characters"""
    out = stem(text)
    out = _expand_greek_unicode(out)
    out = _replace_greek_latin(out)
    return out.lower()


def expand_dashes(text):
    text = _normalize_dashes(text)
    if text.count('-') > 4:
        output = [text]
    else:
        tokens = _dash_tokenize(text)
        output = set(' '.join([c.strip() for c in x if c.strip()])
                     for x in product(*[_expand_token(token)
                                        for token in tokens]))
    return list(output)


def normalize(s):
    s = ''.join(s.split())
    s = ''.join([char for char in s if char not in dashes])
    return s.lower()


def _powerset(n):
    return chain.from_iterable(combinations(range(n), r)
                               for r in range(n+1))


def _normalize_dashes(text):
    out = ''
    for char in text:
        if char in dashes:
            out += '-'
        else:
            out += char
    out = '-'.join([x for x in out.split('-') if x])
    return out


def _expand_token(text):
    tokens = text.split('-')
    if len(tokens) > 5:
        return [text, text.replace('-', '')]
    out = []
    for subset in _powerset(len(tokens) - 1):
        result = tokens[0]
        for i, token in enumerate(tokens[1:]):
            if i in subset:
                result += token
            else:
                result += ' ' + token
        out.append(result)
    return out


def _dash_tokenize(text):
    pattern = re.compile(r'[\w-]+|[^\s\w]')
    matches = re.finditer(pattern, text)
    return [m.group() for m in matches]



def _expand_greek_unicode(text):
    for greek_uni, greek_spelled_out in greek_alphabet.items():
        text = text.replace(greek_uni, greek_spelled_out)
    return text


def _replace_greek_latin(s):
    """Replace Greek spelled out letters with their latin character."""
    for greek_spelled_out, latin in greek_to_latin.items():
        s = s.replace(greek_spelled_out, latin)
    return s


dashes = [chr(0x2212), chr(0x002d)] + [chr(c) for c in range(0x2010, 0x2016)]

greek_alphabet = {
    u'\u0391': 'Alpha',
    u'\u0392': 'Beta',
    u'\u0393': 'Gamma',
    u'\u0394': 'Delta',
    u'\u0395': 'Epsilon',
    u'\u0396': 'Zeta',
    u'\u0397': 'Eta',
    u'\u0398': 'Theta',
    u'\u0399': 'Iota',
    u'\u039A': 'Kappa',
    u'\u039B': 'Lamda',
    u'\u039C': 'Mu',
    u'\u039D': 'Nu',
    u'\u039E': 'Xi',
    u'\u039F': 'Omicron',
    u'\u03A0': 'Pi',
    u'\u03A1': 'Rho',
    u'\u03A3': 'Sigma',
    u'\u03A4': 'Tau',
    u'\u03A5': 'Upsilon',
    u'\u03A6': 'Phi',
    u'\u03A7': 'Chi',
    u'\u03A8': 'Psi',
    u'\u03A9': 'Omega',
    u'\u03B1': 'alpha',
    u'\u03B2': 'beta',
    u'\u03B3': 'gamma',
    u'\u03B4': 'delta',
    u'\u03B5': 'epsilon',
    u'\u03B6': 'zeta',
    u'\u03B7': 'eta',
    u'\u03B8': 'theta',
    u'\u03B9': 'iota',
    u'\u03BA': 'kappa',
    u'\u03BB': 'lamda',
    u'\u03BC': 'mu',
    u'\u03BD': 'nu',
    u'\u03BE': 'xi',
    u'\u03BF': 'omicron',
    u'\u03C0': 'pi',
    u'\u03C1': 'rho',
    u'\u03C3': 'sigma',
    u'\u03C4': 'tau',
    u'\u03C5': 'upsilon',
    u'\u03C6': 'phi',
    u'\u03C7': 'chi',
    u'\u03C8': 'psi',
    u'\u03C9': 'omega',
}

greek_to_latin = {
   'alpha': 'a',
   'Alpha': 'A',
   'beta': 'b',
   'Beta': 'B',
   'gamma': 'c',
   'Gamma': 'C',
   'delta': 'd',
   'Delta': 'D',
}
