"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from torch_geometric.nn import SchNet

from ocpmodels.models.base import BaseModel

import torch
from torch_scatter import scatter
from torch.nn.utils.rnn import pad_sequence

from ocpmodels.models.KAN import KAN
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import (
    conditional_grad,
    get_pbc_distances,
    radius_graph_pbc,
)

# swish implementation
def swish(x):
    return x * torch.sigmoid(x)


class StrainBlock(torch.nn.Module):

    def __init__(
            self,
            out_channels,
            strain_projection_channels,
            num_layers,
            max_atoms,
            act=swish,
    ):
        super(StrainBlock, self).__init__()

        self.max_atoms = max_atoms
        self.act = act
        self.lins = torch.nn.ModuleList()

        # Initializing KAN layers directly
        self.lins.append(KAN([self.max_atoms + 3, strain_projection_channels]))
        for _ in range(num_layers):
            self.lins.append(KAN([strain_projection_channels, strain_projection_channels]))
        self.lins.append(KAN([strain_projection_channels, out_channels]))

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            for layer in lin.layers:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

    def forward(self, x, natoms, feature):
        splits = torch.tensor_split(x, torch.cumsum(natoms, 0)[:-1].cpu())
        x = pad_sequence(splits, batch_first=True)
        x = torch.nn.functional.pad(x, pad=(0, 0, 0, self.max_atoms - x.shape[1], 0, 0), mode='constant', value=0.)

        feature = torch.tensor(feature).to(x.device).to(torch.float32)
        x = torch.cat((x, feature.unsqueeze(-1).repeat(1, 1, x.shape[-1])), dim=1).permute(0, 2, 1)

        for lin in self.lins:
            x = self.act(lin(x))

        return x

@registry.register_model("schnet")
class SchNetWrap(SchNet, BaseModel):
    r"""Wrapper around the continuous-filter convolutional neural network SchNet from the
    `"SchNet: A Continuous-filter Convolutional Neural Network for Modeling
    Quantum Interactions" <https://arxiv.org/abs/1706.08566>`_. Each layer uses interaction
    block of the form:

    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \odot
        h_{\mathbf{\Theta}} ( \exp(-\gamma(\mathbf{e}_{j,i} - \mathbf{\mu}))),

    Args:
        num_atoms (int): Unused argument
        bond_feat_dim (int): Unused argument
        num_targets (int): Number of targets to predict.
        use_pbc (bool, optional): If set to :obj:`True`, account for periodic boundary conditions.
            (default: :obj:`True`)
        regress_forces (bool, optional): If set to :obj:`True`, predict forces by differentiating
            energy with respect to positions.
            (default: :obj:`True`)
        otf_graph (bool, optional): If set to :obj:`True`, compute graph edges on the fly.
            (default: :obj:`False`)
        hidden_channels (int, optional): Number of hidden channels.
            (default: :obj:`128`)
        num_filters (int, optional): Number of filters to use.
            (default: :obj:`128`)
        num_interactions (int, optional): Number of interaction blocks
            (default: :obj:`6`)
        num_gaussians (int, optional): The number of gaussians :math:`\mu`.
            (default: :obj:`50`)
        cutoff (float, optional): Cutoff distance for interatomic interactions.
            (default: :obj:`10.0`)
        readout (string, optional): Whether to apply :obj:`"add"` or
            :obj:`"mean"` global aggregation. (default: :obj:`"add"`)
    """

    def __init__(
        self,
        num_atoms,  # not used
        bond_feat_dim,  # not used
        num_targets,
        use_pbc=True,
        regress_forces=True,
        otf_graph=False,
        hidden_channels=128,
        num_filters=128,
        num_interactions=6,
        num_gaussians=50,
        cutoff=10.0,
        readout="add",
        strain_projection_channels=16,
        num_strain_layers=2,
        strain_final_dim=16,
        max_atoms=0,
    ):
        self.num_targets = num_targets
        self.regress_forces = regress_forces
        self.use_pbc = use_pbc
        self.cutoff = cutoff
        self.otf_graph = otf_graph
        self.max_neighbors = 50
        self.max_atoms = max_atoms
        super(SchNetWrap, self).__init__(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            readout=readout,
        )
        self.strain_block = StrainBlock(strain_final_dim, strain_projection_channels, num_strain_layers, self.max_atoms,)


    @conditional_grad(torch.enable_grad())
    def _forward(self, data):
        z = data.atomic_numbers.long()
        pos = data.pos
        batch = data.batch

        (
            edge_index,
            edge_weight,
            distance_vec,
            cell_offsets,
            _,  # cell offset distances
            neighbors,
        ) = self.generate_graph(data)

        if self.use_pbc:
            assert z.dim() == 1 and z.dtype == torch.long

            edge_attr = self.distance_expansion(edge_weight)

            h = self.embedding(z)
            for interaction in self.interactions:
                h = h + interaction(h, edge_index, edge_weight, edge_attr)

            h = self.lin1(h)
            h = self.act(h)
            h = self.lin2(h)

            batch = torch.zeros_like(z) if batch is None else batch
            h = self.strain_block(h[:, :1], data.natoms, data.feature)
 #           energy = scatter(h, batch, dim=0, reduce=self.readout)
            energy = h.sum(dim=-1)

        else:
            energy = super(SchNetWrap, self).forward(z, pos, batch)
        return energy

    def forward(self, data):
        if self.regress_forces:
            data.pos.requires_grad_(True)
        energy = self._forward(data)

        if self.regress_forces:
            forces = -1 * (
                torch.autograd.grad(
                    energy,
                    data.pos,
                    grad_outputs=torch.ones_like(energy),
                    create_graph=True,
                )[0]
            )
            return energy, forces
        else:
            return energy

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

