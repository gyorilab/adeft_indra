from adeft_indra.mesh import MeshMapper
from indra.databases.hgnc_client import get_hgnc_name
from indra.literature.pubmed_client import get_ids_for_mesh, get_ids_for_gene


_mesh_mapper = MeshMapper()


def get_pmids_for_entity(namespace, id_, major_topic=True):
    if namespace == 'MESH':
        return get_ids_for_mesh(id_, major_topic=major_topic)
    pmids = set()
    mesh_ids = _mesh_mapper.map_to_mesh(namespace, id_,)
    for id_ in mesh_ids:
        pmids.update(get_ids_for_mesh(id_, major_topic=major_topic))
    if namespace == 'HGNC':
        name = get_hgnc_name(id_)
        if name is not None:
            try:
                pmids.update(get_ids_for_gene(name))
            except ValueError:
                pass
    return list(pmids)
