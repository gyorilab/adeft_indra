import logging
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
def get_texts_for_agent_texts(agent_texts):
    ref_cache = SqliteDict(filename=CACHE_PATH, tablename='refs')
    content_cache = SqliteDict(filename=CACHE_PATH, tablename='content')
    texts = []
    for agent_text in agent_texts:
        stmt_ids = get_agent_stmts(agent_text)
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
    text_cache = SqliteDict(filename=CACHE_PATH, tablename='text_cache')
    key = hash((content, frozenset(contains)))
    try:
        text = text_cache[key]
    except KeyError:
        text = universal_extract_text(content, contains)
        text_cache[key] = text
        text_cache.commit()
    return text
