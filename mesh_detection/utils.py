import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import json


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


def add_channel_dim(image, key_points):
    return image[None], key_points


def randomly_shift(image, key_points, max_shift=30):
    shift = np.random.randint(-max_shift, max_shift, size=len(image.shape))

    # shift image
    shape = np.asarray(image.shape)
    start = np.maximum(0, -shift)
    stop = shape - np.maximum(0, shift)
    cropped_image = image[tuple(slice(axis_start, axis_stop) for axis_start, axis_stop in zip(start, stop))]
    shifted_image = np.pad(cropped_image, pad_width=tuple((a, b) for a, b in zip(shape - stop, start)), mode='constant')

    # shift key points
    shifted_key_points = key_points + shift
    valid = np.all((shifted_key_points >= 0) & (shifted_key_points <= shape), 1)
    shifted_key_points[~valid] = np.nan

    return shifted_image, shifted_key_points.astype(np.float32)


def composition(*transformers):
    def apply(*inputs):
        for transformer in transformers:
            inputs = transformer(*inputs)

        return inputs

    return apply


def get_device(x=None):
    if isinstance(x, nn.Module):
        try:
            return next(x.parameters()).device
        except StopIteration:
            raise ValueError('The device could not be determined as the passed model has no parameters.')
    if isinstance(x, torch.Tensor):
        return x.device

    if x is None:
        x = 'cuda' if torch.cuda.is_available() else 'cpu'

    return torch.device(x)


def to_np(x):
    return x.data.cpu().numpy()


def to_torch(x, device=None):
    return torch.from_numpy(x).to(device=get_device(device))


def sequence_to_torch(*inputs, device=None):
    return [to_torch(x, device) for x in inputs]


def show_key_points(key_points):
    ys, xs = key_points.T
    plt.scatter(xs, ys)
