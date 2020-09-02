from adeft_indra.mesh import MeshMapper
from indra.databases.hgnc_client import get_hgnc_name
from indra.literature.pubmed_client import get_ids_for_mesh, get_ids_for_gene


_mesh_mapper = MeshMapper()


def get_pmids_for_entity(namespace, id_, major_topic=True):
    pmids = set()
    if namespace == 'MESH':
        mesh_ids = [id_]
    else:
        mesh_ids = _mesh_mapper.map_to_mesh(namespace, id_)
    if namespace == 'HGNC':
        name = get_hgnc_name(id_)
        if name is not None:
            pmids.update(get_ids_for_gene(name))
    for mesh_id in mesh_ids:
        pmids.update(get_ids_for_mesh(mesh_id, major_topic=major_topic))
    return list(pmids)
