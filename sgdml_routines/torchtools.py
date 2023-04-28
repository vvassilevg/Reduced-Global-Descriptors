#!/usr/bin/python

# MIT License
#
# Copyright (c) 2019-2021 Jan Hermann, Stefan Chmiela
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .utils.desc import Desc
#from .utils.torch_desc import Desc
import time

class GDMLTorchPredict(nn.Module):
    """
    PyTorch version of :class:`~predict.GDMLPredict`. Derives from
    :class:`torch.nn.Module`. Contains no trainable parameters.
    """

    def __init__(self, model, lat_and_inv=None, batch_size=None, max_memory=None):
        """
        Parameters
        ----------
        model : Mapping
            Obtained from :meth:`~train.GDMLTrain.train`.
        lat_and_inv : tuple of :obj:`numpy.ndarray`
            Tuple of 3 x 3 matrix containing lattice vectors as columns and its inverse.
        batch_size : int, optional
            Maximum batch size of geometries for prediction. Calculated from
            :paramref:`max_memory` if not given.
        max_memory : float, optional
            (unit GB) Maximum allocated memory for prediction.
        """

        global _batch_size

        super(GDMLTorchPredict, self).__init__()

        self._log = logging.getLogger(__name__)

        model = dict(model)

        self._dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._lat_and_inv = (
            None
            if lat_and_inv is None
            else (
                torch.tensor(lat_and_inv[0], device=self._dev),
                torch.tensor(lat_and_inv[1], device=self._dev),
            )
        )

        self._sig = int(model['sig'])
        self._c = float(model['c'])
        self._std = float(model.get('std', 1))

        self.n_atoms = model['z'].shape[0]
        self.tril_indices = np.tril_indices(self.n_atoms, k=-1)
        #self.tril_indices = torch.tril_indices(self.n_atoms, self.n_atoms, offset=-1, device=self._dev)

        desc_siz = model['R_desc'].shape[0]
        self.n_train = model['R_desc'].shape[1]
        n_perms, self._n_atoms = model['perms'].shape
        perm_idxs = (
            torch.tensor(model['tril_perms_lin'], device=self._dev)
            .view(-1, n_perms)
            .t()
        )

        self._xs_train, self._Jx_alphas = (
            nn.Parameter(
                xs.repeat(1, n_perms)[:, perm_idxs].reshape(-1, desc_siz),
                requires_grad=False,
            )
            for xs in (
                torch.tensor(model['R_desc'], device=self._dev).t(),
                torch.tensor(np.array(model['R_d_desc_alpha']), device=self._dev),
            )
        )

        # constant memory requirement (bytes): _xs_train and _Jx_alphas
        const_memory = 2 * self._xs_train.nelement() * self._xs_train.element_size()

        if max_memory is None:
            if torch.cuda.is_available():
                max_memory = min(
                    [
                        torch.cuda.get_device_properties(i).total_memory
                        for i in range(torch.cuda.device_count())
                    ]
                )
            else:
                max_memory = int(
                    2 ** 30 * 32
                )  # 32 GB to bytes as default (hardcoded for now...)

        _batch_size = (
            max((max_memory - const_memory) // self._memory_per_sample(), 1)
            if batch_size is None
            else batch_size
        )
        if torch.cuda.is_available():
            _batch_size *= torch.cuda.device_count()

        self.desc = Desc(self.n_atoms, interact_cut_off=model['interact_cut_off'],
            desc_type=model['desc_type'],
            rm_arr=model['rm_arr'],
        )  # NOTE: max processes not set!!

        #self.from_R = self.desc.from_R
        #self.d_desc_from_comp = self.desc.d_desc_from_comp

        self.perm_idxs = perm_idxs
        self.n_perms = n_perms

    def set_alphas(self, R_d_desc, alphas):
        """
        Reconfigure the current model with a new set of regression parameters.
        This is necessary when training the model iteratively.

        Parameters
        ----------
                R_d_desc : :obj:`numpy.ndarray`
                    Array containing the Jacobian of the descriptor for
                    each training point.
                alphas : :obj:`numpy.ndarray`
                    1D array containing the new model parameters.
        """

        dim_d = self.desc.dim
        dim_i = self.desc.dim_i



        # NEW

        # alphas = torch.from_numpy(alphas).to(self._dev)
        # alphas = alphas.reshape(-1, self.n_atoms, 3)

        # if self.R_d_desc is None:
        #     self.R_d_desc = torch.from_numpy(R_d_desc).to(self._dev)

        # i, j = self.tril_indices
        # xs = torch.einsum('kji,kji->kj', self.R_d_desc, alphas[:, j, :] - alphas[:, i, :])

        # del alphas


        # NEW


        if self.desc.desc_type != 1:
            
            R_d_desc_full = self.desc.d_desc_from_comp(R_d_desc).reshape(self.n_train, dim_d, self.n_atoms, 3).reshape(
                self.n_train, dim_d, -1
            )
            R_d_desc_alpha = np.einsum(
                #'kji,ki->kj', R_d_desc_full, torch.from_numpy(alphas).reshape(self.n_train, -1) #.to(R_d_desc_full.device)
                'kji,ki->kj', R_d_desc_full, alphas.reshape(self.n_train, -1) 
            )
            R_d_desc_alpha = torch.from_numpy(R_d_desc_alpha).to(self._dev)
            del R_d_desc_full
        else:
            
            R_d_desc_alpha = self.desc.d_desc_dot_vec(R_d_desc, alphas.reshape(-1, dim_i))
            R_d_desc_alpha = torch.from_numpy(R_d_desc_alpha).to(self._dev)
 
        #xs = torch.from_numpy(R_d_desc_alpha).to(self._dev)
        xs = R_d_desc_alpha

        self._Jx_alphas = nn.Parameter(
            xs.repeat(1, self.n_perms)[:, self.perm_idxs].reshape(-1, dim_d),
            requires_grad=False,
        )

    def _memory_per_sample(self):

        # peak memory:
        # N * a * a * 3
        # N * d * 2
        # N * n_perms*N_train * (d+4)

        dim_d = self._xs_train.shape[1]

        total = (dim_d * 2 + self.n_atoms) * 3
        total += dim_d * 2
        total += self._xs_train.shape[0] * (dim_d + 4)

        return total * self._xs_train.element_size()

    def _batch_size(self):
        return _batch_size

    def _forward(self, Rs):

        sig = self._sig
        q = np.sqrt(5) / sig

        diffs = Rs[:, :, None, :] - Rs[:, None, :, :]  # N, a, a, 3
        if self._lat_and_inv is not None:
            diffs_shape = diffs.shape
            # diffs = self.desc.pbc_diff(diffs.reshape(-1, 3), self._lat_and_inv).reshape(
            #    diffs_shape
            # )

            lat, lat_inv = self._lat_and_inv

            if lat.device != Rs.device:
                lat = lat.to(Rs.device)
                lat_inv = lat_inv.to(Rs.device)

            diffs = diffs.reshape(-1, 3)

            c = lat_inv.mm(diffs.t())
            diffs -= lat.mm(c.round()).t()

            diffs = diffs.reshape(diffs_shape)

        #dists = diffs.norm(dim=-1)  # N, a, a

        i, j = self.tril_indices

        if self.desc.desc_type == 1:
            dists = diffs.norm(dim=-1)  # N, a, a
            xs = 1 / dists[:, i, j]  # R_desc # N, d
            del dists
        else:
            xs, d_xs = self.desc.from_R(Rs, self._lat_and_inv, USE_CUDA=True)
            d_xs = self.desc.d_desc_from_comp(d_xs) #[0]


            xs = torch.from_numpy(xs).to(Rs.device)
            d_xs = torch.from_numpy(d_xs).to(Rs.device)
       

        # current:
        # diffs: N, a, a, 3
        # dists: N, a, a
        # xs: # N, d

        #del dists

        # current:
        # diffs: N, a, a, 3
        # xs: # N, d

        #x_diffs = (q * xs)[:, None, :] - q * self._xs_train  # N, n_perms*N_train, d
        if xs.dim() == 1:
            xs = xs.reshape(diffs.size()[0], xs.size()[0])
        x_diffs = q * (xs[:, None, :] - self._xs_train)  # N, n_perms*N_train, d
        x_dists = x_diffs.norm(dim=-1)  # N, n_perms*N

        exp_xs = 5.0 / (3 * sig ** 2) * torch.exp(-x_dists)  # N, n_perms*N_train

        # dot_x_diff_Jx_alphas = (x_diffs * self._Jx_alphas).sum(dim=-1)
        dot_x_diff_Jx_alphas = torch.einsum(
            'ijk,jk->ij', x_diffs, self._Jx_alphas
        )  # N, n_perms*N_train
        exp_xs_1_x_dists = exp_xs * (1 + x_dists)  # N, n_perms*N_train

        # F1s_x = ((exp_xs * dot_x_diff_Jx_alphas)[..., None] * x_diffs).sum(dim=1)
        # F2s_x = exp_xs_1_x_dists.mm(self._Jx_alphas)

        # Fs_x = ((exp_xs * dot_x_diff_Jx_alphas)[..., None] * x_diffs).sum(dim=1)
        Fs_x = torch.einsum(
            'ij,ij,ijk->ik', exp_xs, dot_x_diff_Jx_alphas, x_diffs
        )  # N, d

        # current:
        # diffs: N, a, a, 3
        # xs: # N, d
        # x_diffs: # N, n_perms*N_train, d
        # x_dists: # N, n_perms*N_train
        # exp_xs: # N, n_perms*N_train
        # dot_x_diff_Jx_alphas: N, n_perms*N_train
        # exp_xs_1_x_dists: N, n_perms*N_train
        # Fs_x: N, d

        del exp_xs
        del x_diffs

        Fs_x -= exp_xs_1_x_dists.mm(self._Jx_alphas)  # N, d

        # current:
        # diffs: N, a, a, 3
        # xs: # N, d
        # x_dists: # N, n_perms*N
        # dot_x_diff_Jx_alphas: N, n_perms*N
        # exp_xs_1_x_dists: N, n_perms*N
        # Fs_x: N, d
    
        # Fs_x = (F1s_x - F2s_x) * (xs ** 3)
        if self.desc.desc_type == 1:
            Fs_x *= xs ** 3
            diffs[:, i, j, :] *= Fs_x[..., None]
            diffs[:, j, i, :] *= Fs_x[..., None]

            Fs = diffs.sum(dim=1) * self._std
            del diffs
        else: 
            del diffs
            conf_shape = list(Fs_x.size())[0]

            Fs = [d_xs[kk,:,:].T.mm(torch.reshape(Fs_x[kk,:], (-1,1))) for kk in range(conf_shape)]         
        
            Fs = torch.stack(Fs) * self._std
            Fs = torch.reshape(Fs, (conf_shape, -1))
        


        # Es = (exp_xs_1_x_dists * dot_x_diff_Jx_alphas).sum(dim=-1) / q
        Es = torch.einsum('ij,ij->i', exp_xs_1_x_dists, dot_x_diff_Jx_alphas) / q
        Es *= self._std
        Es += self._c

        return Es, Fs

    def forward(self, Rs):
        """
        Predict energy and forces for a batch of geometries.

        Parameters
        ----------
        Rs : :obj:`torch.Tensor`
            (dims M x N x 3) Cartesian coordinates of M molecules composed of N atoms

        Returns
        -------
        E : :obj:`torch.Tensor`
            (dims M) Molecular energies
        F : :obj:`torch.Tensor`
            (dims M x N x 3) Nuclear gradients of the energy
        """

        global _batch_size

        assert Rs.dim() == 3
        assert Rs.shape[1:] == (self._n_atoms, 3)

        dtype = Rs.dtype
        Rs = Rs.double()

        while True:
            try:
                Es, Fs = zip(
                    *map(self._forward, DataLoader(Rs, batch_size=_batch_size))
                )
                #Es, Fs = zip(
                #    *map(self._forward, DataLoader(Rs, batch_size=_batch_size, multiprocessing_context='spawn', num_workers=1))#, multiprocessing_context='spawn'))
                #)
 
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    if _batch_size > (torch.cuda.device_count() + 1):

                        import gc
                        gc.collect()

                        torch.cuda.empty_cache()

                        _batch_size -= 1

                    else:
                        self._log.critical(
                            'Could not allocate enough memory to evaluate model, despite reducing batch size.'
                        )
                        print()
                        sys.exit()
                else:
                    raise e
            else:
                break

        return torch.cat(Es).to(dtype), torch.cat(Fs).to(dtype)
