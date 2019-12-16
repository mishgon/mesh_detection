import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm_notebook as tqdm

from mesh_detection.utils import to_np, sequence_to_torch
from mesh_detection.model import predict


def criterion(output, target):
    not_nan = ~torch.isnan(target)
    nothing = torch.zeros(1, requires_grad=True)[0]
    return F.mse_loss(output[not_nan], target[not_nan]) if not_nan.any() else nothing


def validator(val_ids, load_image, load_key_points, metrics):
    def validate(model):
        metrics_values = {name: [] for name in metrics.keys()}
        for i in val_ids:
            prediction = predict(load_image(i), model)
            target = load_key_points(i)
            for name, metric in metrics.items():
                metrics_values[name].append(metric(prediction, target))

        return {name: np.nanmean(values, axis=0) for name, values in metrics_values.items()}

    return validate


def train(model, batch_iter, optimizer, criterion, n_epochs, validate=None):
    train_losses = []
    val_metrics = []
    for epoch in tqdm(range(n_epochs)):
        # training
        epoch_losses = []
        for images, key_points in batch_iter():
            x, target = sequence_to_torch(images, key_points, device=model)
            loss = criterion(model(x), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(to_np(loss))

        train_losses.append(np.mean(epoch_losses, axis=0))

        # validation
        if validate is not None:
            val_metrics.append(validate(model))

    if len(val_metrics):
        val_metrics = {name: [epoch_metric[name] for epoch_metric in val_metrics] for name in val_metrics[0].keys()}

    return model, train_losses, val_metrics



