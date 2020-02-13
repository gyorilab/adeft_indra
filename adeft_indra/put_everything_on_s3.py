import os
import boto3
import tempfile
import logging
from sqlalchemy import text
from multiprocessing.pool import ThreadPool

logger = logging.getLogger()

from indra_db.util import get_primary_db

BUCKET = 'bigmech'


def get_all_content_ids():
    db = get_primary_db()
    query = 'SELECT id from text_content'
    res = db.session.execute(text(query))
    return [x[0] for x in res.fetchall()]


def stream_text_content(content_ids, batch_size=1000000):
    db = get_primary_db()
    query = """SELECT
                   text_ref_id, source, text_type, content
               FROM
                   text_content
               WHERE
                   id in :tcids
            """
    for i in range(0, len(content_ids), batch_size):
        tcid_batch = content_ids[i:i+batch_size]
        res = db.session.execute(text(query), {'tcids': tuple(tcid_batch)})
        yield res


client = boto3.client('s3')


def upload_file(x):
    key, content = x
    with tempfile.NamedTemporaryFile() as temp:
        with open(temp.name, 'wb') as f:
            f.write(content)
            client.upload_file(temp.name, BUCKET, key)


def content_key_generator(batch):
    for text_ref_id, source, text_type, content in batch:
        key = os.path.join('content', str(text_ref_id), source, text_type,
                           f'content.zlib')
        content = content.tobytes()
        yield (key, content)


def do_it():
    content_stream = stream_text_content(get_all_content_ids())
    for i, batch in enumerate(content_stream):
        logger.info(f'working on batch: {i}')
        with ThreadPool(8) as pool:
            pool.map(upload_file, content_key_generator(batch))


if __name__ == '__main__':
    do_it()
