import logging
import requests


logger = logging.getLogger(__name__)


def gilda_ground(agent_text):
    """Use the grounding service to produce groundings for agent text

    Parameters
    ----------
    agent_text : str
        Text of agent to ground

    Returns
    -------
    response : dict
    """
    grounding_service_ip = '0.0.0.0:8001'
    grounding_service_url = f'http://{grounding_service_ip}/ground'
    response = requests.post(grounding_service_url,
                             json={'text': agent_text})
    if response.status_code != 200:
        raise RuntimeError(f'Received response with code'
                           '{response.status_code}')
    result = response.json()
    return result
    if not result:
        output = (None, None, None)
    else:
        term = result[0]['term']
        output = term['db'], term['id'], term['entry_name']
    return output


def make_grounding_map(texts, ground=gilda_ground):
    groundings = {text: ground(text) for text in texts}
    grounding_map = {}
    names = {}
    for text, (db, id_, name) in groundings.items():
        if id_ is not None:
            grounding_map[text] = db + ':' + id_
            names[db + ':' + id_] = name
        else:
            grounding_map[text] = 'ungrounded'
    return grounding_map, names
