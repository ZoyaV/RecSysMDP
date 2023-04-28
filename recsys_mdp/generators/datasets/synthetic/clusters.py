from __future__ import annotations

import numpy as np
from numpy.random import Generator


def generate_clusters(
        rng: Generator,
        n_clusters: int | list[int],
        n_dims: int,
        n_dissimilar_dims_required: int = 3,
        min_dim_delta: float = 0.4,
        min_l2_dist: float = 0.1,
        max_tries: int = 10000,
):
    n_centers = n_clusters if isinstance(n_clusters, int) else len(n_clusters)
    centers = _generate_distant_samples(
        rng, n_centers, n_dims,
        n_dissimilar_dims_required=n_dissimilar_dims_required,
        min_dim_delta=min_dim_delta,
        min_l2_dist=min_l2_dist,
        max_tries=max_tries
    )

    if isinstance(n_clusters, int):
        return centers

    samples = []
    for center, n_sub_centers in zip(centers, n_clusters):
        clustered_sub_clusters = _generate_distant_samples(
            rng, n_sub_centers, n_dims,
            n_dissimilar_dims_required=n_dissimilar_dims_required,
            min_dim_delta=min_dim_delta / 2,
            min_l2_dist=min_l2_dist / 2,
            w=0.2,
            center=center,
            max_tries=max_tries
        )
        samples.extend(clustered_sub_clusters)
    # samples around center may be out of [0, 1] range, clip them
    samples = np.clip(0, 1, np.array(samples))

    return samples


def _generate_distant_samples(
        rng: Generator,
        n_samples: int,
        ndims: int,
        n_dissimilar_dims_required: int,
        min_dim_delta: float,
        min_l2_dist: float,
        w: float = 1,
        center: float = 0.5,
        max_tries: int = 10000,
):
    for _ in range(max_tries):
        samples = rng.uniform(size=(n_samples, ndims))
        if _check_dissimilarity(samples, min_dim_delta, n_dissimilar_dims_required, min_l2_dist):
            return _shift_rescale(samples, w, center)

    raise RuntimeError(f'Could not generate samples in {max_tries} tries')


def _check_dissimilarity(samples, min_dim_delta, n_dissimilar_dims_required, min_l2_dist):
    for i in range(samples.shape[0] - 1):
        # (right tail size, ndims)
        diff_with_tail = np.abs(samples[i + 1:] - samples[i])

        # (right tail size,) - count how many dimensions dissimilar to those of the `sample`
        n_dissimilar_dims = np.count_nonzero(diff_with_tail >= min_dim_delta, axis=-1)
        dims_dissimilarity_failed = n_dissimilar_dims < n_dissimilar_dims_required
        if np.any(dims_dissimilarity_failed):
            # there's a pair to the `sample` which is too similar; next, please
            return False

        # (right tail size,) - squared distances to the `sample`
        sq_distance = np.sum(diff_with_tail ** 2, axis=-1)
        l2_distance_failed = sq_distance < min_l2_dist ** 2
        if np.any(l2_distance_failed):
            # there's a pair to the `sample` which is too close in terms of L2 distance
            return False
    return True


def _shift_rescale(samples, w, center, from_w=1.0, from_center=0.5):
    unit_cube_samples = (samples - from_center) / from_w
    return unit_cube_samples * w + center


def _sort_lexicographically(arr):
    return arr[np.lexsort(np.fliplr(arr).T)]


def _test():
    seed = 123
    samples = generate_clusters(
        np.random.default_rng(seed),
        n_clusters=[2, 3, 4, 1],
        n_dims=5,
        n_dissimilar_dims_required=1,
        min_dim_delta=0.3,
        min_l2_dist=0.3,
        max_tries=100000
    )

    samples = _sort_lexicographically(samples)
    with np.printoptions(precision=1):
        print(samples)

        similarity = 1 - np.sqrt(
            np.sum((samples[:, None] - samples[None, :]) ** 2, axis=-1)
        ) / np.sqrt(samples.shape[1]) / 2
        # print(similarity)

    from matplotlib import pyplot as plt
    plt.imshow(similarity)
    plt.show()


if __name__ == '__main__':
    _test()
