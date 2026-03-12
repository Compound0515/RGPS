import numpy as np
from dscribe.descriptors import SOAP


def compute_soap_descriptors_core(frames, elements, soap_config):
    soap = SOAP(
        species=elements,
        periodic=True,
        r_cut=soap_config.get("r_cut", 6.0),
        n_max=soap_config.get("n_max", 8),
        l_max=soap_config.get("l_max", 6),
        average="inner",
        sparse=False,
    )
    return soap.create(frames)


def select_fps_core(features, n_select=10):
    """Farthest Point Sampling"""
    n_total = features.shape[0]
    if n_select >= n_total:
        return list(range(n_total))

    centroid = np.mean(features, axis=0)
    diff_center = features - centroid
    dist_sq_center = np.sum(diff_center**2, axis=1)
    first = int(np.argmax(dist_sq_center))
    selected = [first]

    diff = features - features[first]
    min_dists_sq = np.sum(diff**2, axis=1)

    for _ in range(1, n_select):
        idx = int(np.argmax(min_dists_sq))
        selected.append(idx)

        diff = features - features[idx]
        dist_sq = np.sum(diff**2, axis=1)
        np.minimum(min_dists_sq, dist_sq, out=min_dists_sq)

    return selected
