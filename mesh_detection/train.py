import numpy as np

from mesh_detection.utils import to_np, sequence_to_torch
from mesh_detection.model import predict


def validator(images, key_points, metrics):
    def validate(model):
        predictions = [predict(img, model) for img in images]
        average_metrics = {}
        for name, metric in metrics.items():
            average_metrics[name] = np.mean([metric(kps, pred) for kps, pred in zip(key_points, predictions)])

        return average_metrics

    return validate


def train(model, batch_iter, optimizer, criterion, n_epochs, validate):
    train_losses = []
    val_metrics = []
    for epoch in range(n_epochs):
        # training
        epoch_losses = []
        for images, key_points in batch_iter:
            x, target = sequence_to_torch(images, key_points, device=model)
            loss = criterion(model(x), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(to_np(loss))

        train_losses.append(np.mean(epoch_losses, axis=0))

        # validation
        val_metrics.append(validate(model))

    if len(val_metrics):
        val_metrics = {name: [epoch_metric[name] for epoch_metric in val_metrics] for name in val_metrics[0].keys()}

    return model, train_losses, val_metrics



