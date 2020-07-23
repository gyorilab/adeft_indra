import random
from sqlalchemy import text
from collections import defaultdict

from indra.util import batch_iter
from indra.literature.adeft_tools import universal_extract_text

from indra_db.util import get_primary_db


def collect_best_content(content_ids):
    priority = {'fulltext': 2, 'abstract': 1, 'title': 0}
    seen_text_refs = {}
    ref_dict = {}
    for id_, text_ref_id, source, format_, text_type in content_ids:
        new_identifier = (id_, text_ref_id, source, format_, text_type)
        if text_ref_id not in seen_text_refs:
            seen_text_refs[text_ref_id] = new_identifier
            ref_dict[text_ref_id] = new_identifier
        else:
            # update if we find text_type with higher priority for
            # a given text_ref
            old_identifier = seen_text_refs[text_ref_id]
            old_text_type = old_identifier[4]
            if priority[text_type] > priority[old_text_type]:
                seen_text_refs[text_ref_id] = new_identifier
                ref_dict[text_ref_id] = new_identifier
    return list(ref_dict.values())


def get_best_content():
    db = get_primary_db()
    query = '''SELECT
                   id, text_ref_id, source, format, text_type
               FROM
                   text_content
            '''
    res = db.session.execute(text(query))
    best_content = collect_best_content(res.fetchall())
    return best_content


def partition_content(content_ids, bytes_per_partition,
                      max_partition_size):
    size_map = {'title': 250, 'abstract': 3000, 'fulltext': 200000}
    content_ids.sort(key=lambda x: x[0])
    output = defaultdict(list)
    current_bucket = 0
    current_bytes = 0
    for content_id in content_ids:
        effective_size = size_map[content_id[-1]]
        if (current_bytes + effective_size > bytes_per_partition or
            len(output[current_bucket]) + 1 > max_partition_size):
            current_bucket += 1
            output[current_bucket].append(content_id[0])
            current_bytes = effective_size
        else:
            output[current_bucket].append(content_id[0])
            current_bytes += effective_size
    return dict(output)


def get_texts(tcids):
    db = get_primary_db()
    tcids = tuple(tcids)
    query = """SELECT
                   id, content
               FROM
                   text_content
               WHERE
                   id in :tcids
            """
    res = db.session.execute(text(query), {'tcids': tcids})
    return [(tcid, memview.tobytes()) for tcid, memview in res.fetchall()]
