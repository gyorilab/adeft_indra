import os
import boto3
import tempfile
import logging
from sqlalchemy import text
from multiprocessing.pool import ThreadPool

logger = logging.getLogger()

from indra.util import batch_iter
from indra_db.util import get_primary_db

BUCKET = 'bigmech'


def stream_text_content(batch_size=1000000):
    db = get_primary_db()
    query_for_ids = 'SELECT id from text_content WHERE id >= 6000000'
    id_stream = db.session.execute(text(query_for_ids))
    query_for_content = """SELECT
                               text_ref_id, source, format, text_type, content
                           FROM
                               text_content
                           WHERE
                               id in :tcids
            """
    for tcid_batch in batch_iter(id_stream, batch_size):
        tcids = tuple(tcid[0] for tcid in tcid_batch)
        logger.info('gathering content')
        res = db.session.execute(text(query_for_content), {'tcids': tcids})
        yield res


client = boto3.client('s3')


def upload_file(x):
    key, content = x
    with tempfile.NamedTemporaryFile() as temp:
        with open(temp.name, 'wb') as f:
            f.write(content)
        client.upload_file(temp.name, BUCKET, key)


def content_key_generator(batch):
    for text_ref_id, source, format_, text_type, content in batch:
        key = os.path.join('content', str(text_ref_id), source, format_,
                           text_type, f'content.zlib')
        content = content.tobytes()
        yield (key, content)


def do_it():
    content_stream = stream_text_content()
    for i, batch in enumerate(content_stream):
        logger.info(f'uploading content for batch: {i}')
        with ThreadPool(512) as pool:
            pool.map(upload_file, content_key_generator(batch))


if __name__ == '__main__':
    do_it()
