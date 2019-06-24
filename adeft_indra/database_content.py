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
    content_cache = SqliteDict(filename=CACHE_PATH, tablename='content')
    for agent_text, stmts in agent_stmts:
        ref_dict, text_dict = cs.get_text_content_from_stmt_ids(stmts)
        content_cache[agent_text] = {'ref_dict': ref_dict,
                                     'text_dict': text_dict}
    content_cache.commit()


@ensure_adeft_indra_folder
def get_texts_for_agent_texts(agent_texts):
    content_cache = SqliteDict(filename=CACHE_PATH, table_name='content')
    texts = set()
    for agent_text in agent_texts:
        try:
            _, text_dict = content_cache[agent_text]
        except KeyError:
            fill_content_cache([agent_text])
            _, text_dict = content_cache[agent_text]
        texts.update(text_dict.values())
    texts = [universal_extract_text(text) for text in texts]
    return [text for text in texts if text]
