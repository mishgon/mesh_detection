import numpy as np


def pad_to_largest(images):
    resulting_shape = tuple(map(max, zip(*[img.shape for img in images])))
    return [pad_to_shape(img, resulting_shape) for img in images]


def pad_to_shape(image, shape):
    image_shape, shape = np.asarray(image.shape), np.asarray(shape)
    assert len(image_shape) == len(shape) and np.all(shape - image_shape >= 0), (image_shape, shape)

    def split_to_equal(n: int):
        return (n // 2, n // 2 + 1) if n % 2 else (n // 2, n // 2)

    pad_width = list(map(split_to_equal, shape - image_shape))
    return np.pad(image, pad_width, mode='constant')


def randomly_flip(image, key_points):
    if np.random.binomial(1, .5):
        image = np.flip(image, 1)
        key_points[:, 0] = image.shape[1] - key_points[:, 0]

    return image, key_points


def add_channel_dim(image, key_points):
    return image[None], key_points


def identity(x):
    return x


def composition(*transformers):
    def apply(*inputs):
        for transformer in transformers:
            inputs = transformer(*inputs)

        return inputs

    return apply
