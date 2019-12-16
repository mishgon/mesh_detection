from copy import copy
import numpy as np
from mesh_detection.utils import pad_to_largest, composition


def batch_iterator(load_image, load_key_points, ids, batch_size, *transformers):
    def iterate_batches():
        shuffled_ids = copy(ids)
        np.random.shuffle(shuffled_ids)

        apply = composition(*transformers)

        for start in range(0, len(shuffled_ids), batch_size):
            batch_ids = ids[start:start + batch_size]
            images, key_points = zip(*[apply(load_image(i), load_key_points(i)) for i in batch_ids])
            yield np.stack(pad_to_largest(images)), np.stack(key_points)

    return iterate_batches

