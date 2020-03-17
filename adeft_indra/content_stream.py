from indra.util import batch_iter
from indra_db.util import get_primary_db

def get_all_trids():
    db = get_primary_db()
    query = 'SELECT id from text_ref'
    res = db.session.execute(sqltext(query))
    return set(x[0] for x in res.fetchall())

def create_batches(batch_size=1000000, random_state=0):
    trids = _get_all_trids()
    trids = [x[0] for x in trids]
    random.seed(random_state)
    random.shuffle(trids)
    batch_dict = {}
    for i, batch in enumerate(batch_iter(trids, batch_size)):
        batch_dict[f'batch_{i}'] = list(batch)
    return batch_dict

