import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from mesh_detection.utils import to_np, to_torch


def predict(image, model):
    return to_np(model(to_torch(image[None, None], device=model)))[0]


class BitNet(nn.Module):
    def __init__(self, structure, n_points, conv_block, pooling):
        """
        Parameters
        ----------
        structure: sequence of sequences
            see Examples section.
        n_points: int
            number of points to predict.
        conv_block: Callable
            ``conv_block(in_channels, out_channels, **kwargs)`` returns a convolution block (basically a convolution but
            maybe a residual block)

        Examples
        --------
        >>> stucture = [
        >>>     [[1, 16, 16],           [16, 32, 32]],
        >>>         [[16, 32, 32],      [32, 64, 64]],
        >>>             [[32, 64, 64],  [64, 128, 128]],
        >>>                 [64, 128, 128]
        >>> ]
        >>> bit_net = BitNet(structure, 1)
        """
        super().__init__()

        def build_conv_seq(channels_seq, final_activation=None, **kwargs):
            conv_seq = []
            for in_channels, out_channels in zip(channels_seq[:-2], channels_seq[1:-1]):
                conv_seq.append(conv_block(in_channels, out_channels, **kwargs))
                conv_seq.append(nn.ReLU())

            conv_seq.append(conv_block(channels_seq[-2], channels_seq[-1]), **kwargs)
            if final_activation is not None:
                conv_seq.append(final_activation())

            return nn.Sequential(*conv_seq)

        self.conv_seqs1 = nn.ModuleList([build_conv_seq(level[0], nn.ReLU) for level in structure[:-1]])
        self.poolings = nn.ModuleList([pooling() for _ in structure[:-1]])
        self.lowest_conv_seq = build_conv_seq([*structure[-1], n_points])
        self.conv_seqs2 = nn.ModuleList([
            build_conv_seq([n_points * n_channels for n_channels in [*level[1], 1]], groups=n_points)
            for level in structure[:-1]
        ])

    @staticmethod
    def softargmax(x):
        """
        Parameters
        ----------
        x: torch.tensor
            Tensor of shape ``(batch_size, n_points, *spatial)`` with logits.

        Returns
        -------
        points: torch.tensor
            Tensor of shape ``(batch_size, n_points, len(spatial))`` with points.
        """
        exp_L = torch.exp(x)

        batch_size, n_points = x.size()[:2]
        size = x.size()[2:]
        sm.reshape((*sm.size(), 1))
        if len(size) == 2:
            H, W = size
            H_coords = torch.arange(H, dtype=torch.float)
            W_coords = torch.arange(W, dtype=torch.float)
            coords = torch.stack([
                H_coords.repeat(W, 1).T.reshape(-1),
                W_coords.repeat(H)
            ]).T

            div = torch.sum(exp_L, dim=(2, 3))

        else:
            H, W, D = size
            H_coords = torch.arange(H, dtype=torch.float)
            W_coords = torch.arange(W, dtype=torch.float)
            D_coords = torch.arange(D, dtype=torch.float)
            coords = torch.stack([
                H_coords.repeat(W * D, 1).T.reshape(-1),
                W_coords.repeat(D, 1).T.reshape(-1).repeat(H),
                D_coords.repeat(H * W)
            ]).T

            div = torch.sum(exp_L, dim=(2, 3, 4))

        samx = torch.floor((exp_L @ coords) / div.reshape((*div.size(), 1))).type(torch.LongTensor)

        return samx


    @staticmethod
    def extract_patches(feature_map, starts, patch_size):
        """
        Parameters
        ----------
        feature_map: torch.tensor
            Tensor of shape ``(batch_size, n_channels, *spatial)`` with logits.
        starts: torch.LongTensor
            Tensor of shape ``(batch_size, n_points, len(spatial)`` with start spatial indices of patches.
        patch_size: int sequence of ints
            Patch size - int ot sequence of ``len(spatial)`` ints.

        Returns
        -------
        patches: torch.tensor
            Tensor of shape ``(batch_size, n_points * n_channels, *patch_size)``.
        """
        batch_size, n_channels, *spatial = feature_map.shape
        patch_size = np.broadcast_to(patch_size, len(spatial))
        bbox_indices = torch.stack(torch.meshgrid(*map(torch.arange, patch_size))).to(starts)
        bbox_indices = bbox_indices + starts[(..., *len(spatial) * [None])]  # of shape (bs, np, len(sp), *ps)
        spatial_indices = [indices.squeeze(2).unqueeze(1) for indices in torch.split(bbox_indices, 1, 2)]
        batch_indices = torch.arange(batch_size)[(slice(None), *(2 + len(spatial)) * [None])]
        channel_indices = torch.arange(n_channels)[(None, slice(None), *(1 + len(spatial)) * [None])]

        # get patches tensor of shape (bs, nc, np, *ps)
        feature_map = torch.unsqueeze(feature_map, 2)
        patches = feature_map[(batch_indices, channel_indices, 0, *spatial_indices)]
        patches = torch.cat([channel.squeeze(1) for channel in torch.split(patches, 1, 1)], 1)
        return patches

    def forward(self, x):
        # contracting path
        feature_maps = []
        for conv_seq, pool in zip(self.conv_seqs1, self.poolings):
            x = conv_seq(x)
            feature_maps.append(x.clone())
            x = pool(x)

        # points predicting path
        points = self.softargmax(self.lowest_conv_seq(x))
        for conv_seq, fm, pool in reversed(list(zip(self.conv_seqs2, feature_maps, self.poolings))):
            points = points.long() * torch.tensor(pool.kernel_size).to(points).long()
            x = self.extract_patches(fm, points, pool.kernel_size)
            points = points + self.softargmax(conv_seq(x))

        return points





