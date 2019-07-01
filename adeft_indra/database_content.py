import logging
import hashlib
from sqlitedict import SqliteDict


from indra_db.util import content_scripts as cs
from indra.literature.adeft_tools import universal_extract_text

from adeft_indra.locations import ensure_adeft_indra_folder, CACHE_PATH


sqlitedict_logger = logging.getLogger('sqlitedict')
sqlitedict_logger.setLevel(logging.WARNING)


@ensure_adeft_indra_folder
def fill_statement_cache(pattern, filter_genes=False):
    """Fill cache with lists of stmt ids for shortforms matching pattern

    Replaces data in existing cache.
    """
    if isinstance(pattern, str):
        query = cs.get_stmts_with_agent_text_like
    elif isinstance(pattern, list):
        query = cs.get_stmts_with_agent_text_in
    stmt_dict = query(pattern, filter_genes)
    cache = SqliteDict(filename=CACHE_PATH, tablename='stmts')
    for agent_text, stmts in stmt_dict.items():
        cache[agent_text] = stmts
    cache.commit()


@ensure_adeft_indra_folder
def get_agent_stmts(agent_text):
    """Get raw stmt ids for given agent text

    Uses cached results if they exist
    """
    cache = SqliteDict(filename=CACHE_PATH, tablename='stmts')
    try:
        stmts = cache[agent_text]
    except KeyError:
        stmts = cs.get_stmts_with_agent_text_like(agent_text)[agent_text]
        cache[agent_text] = stmts
        cache.commit()
    return stmts


@ensure_adeft_indra_folder
def fill_content_cache(agent_texts=None):
    """Fill cache with mappings to get text content from stmt_ids

    Consists of maps from statement ids to text ref ids and tex_ref
    ids to text content.

    Parameters
    ----------
    agent_text : Optional[list]
        List of agent texts to get text content mappings for.
        If None, get mappings for every statement in statement cache
    """
    if agent_texts is None:
        stmts_cache = SqliteDict(filename=CACHE_PATH, tablename='stmts')
        agent_stmts = stmts_cache.items()
    else:
        agent_stmts = [(agent_text, get_agent_stmts(agent_text))
                       for agent_text in agent_texts]
    ref_cache = SqliteDict(filename=CACHE_PATH, tablename='refs')
    content_cache = SqliteDict(filename=CACHE_PATH, tablename='content')
    for agent_text, stmts in agent_stmts:
        ref_dict, text_dict = cs.get_text_content_from_stmt_ids(stmts)
        for stmt_id, ref in ref_dict.items():
            ref_cache[stmt_id] = ref
            ref_cache.commit()
            if ref is not None and text_dict[ref]:
                content_cache[ref] = text_dict[ref]
                content_cache.commit()


@ensure_adeft_indra_folder
def get_plaintexts_for_agent_texts(agent_texts):
    """Get plaintexts for given agent texts

    Gathers all text content with at least one statement with agent text
    from the input list. Extracts plaintext from xml and filters to only
    paragraphs containing at least one of the agent texts.

    Parameters
    ----------
    agent_texts : list of str
        List of agent texts

    Returns
    -------
    texts : list of str
        List of plaintexts containing agent texts
    """
    texts_cache = SqliteDict(filename=CACHE_PATH, tablename='texts')
    key = ':'.join(sorted(agent_texts))
    try:
        texts = texts_cache[key]
    except KeyError:
        ref_cache = SqliteDict(filename=CACHE_PATH, tablename='refs')
        content_cache = SqliteDict(filename=CACHE_PATH, tablename='content')

        stmt_ids = set()
        for agent_text in agent_texts:
            stmt_ids.update(get_agent_stmts(agent_text))
        texts = []
        for stmt_id in stmt_ids:
            try:
                ref = ref_cache[stmt_id]
            except KeyError:
                fill_content_cache([agent_text])
            ref = ref_cache[stmt_id]
            if ref is not None:
                content = content_cache.get(ref)
                if content:
                    texts.append(universal_extract_text_cached(content,
                                                               agent_texts))
        texts = [text for text in texts if text]
        texts_cache[key] = texts
        texts_cache.commit()
    return texts


def universal_extract_text_cached(content, contains):
    """Returns plaintext for either elsevier or pubmed xml

    Caches results in sqlitedict

    Parameters
    ----------
    content : str
        NLM XML, Elsevier XML, or Plaintext

    contains : list of str
        Filter to only paragraphs containing one of these strings

    Returns
    -------
    text : str
        Plaintext of input, filtered to only paragraphs containing
        one of the texts in contains. Returns the parameter `content`
        if it is not an NLM or Elsevier XML.
    """
    text_cache = SqliteDict(filename=CACHE_PATH, tablename='text_cache')
    if not content:
        return None
    hash_ = hashlib.md5(content.encode()).hexdigest()
    key = hash_ + ':' + ':'.join(sorted(contains))
    try:
        text = text_cache[key]
    except KeyError:
        text = universal_extract_text(content, contains)
        text_cache[key] = text
        text_cache.commit()
    return text
