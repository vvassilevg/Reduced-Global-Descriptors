#!/usr/bin/python

# MIT License
#
# Copyright (c) 2020-2021 Stefan Chmiela
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

from functools import partial
import inspect
import multiprocessing as mp

import numpy as np
import scipy as sp
import timeit
import collections

from .. import DONE, NOT_DONE
from ..utils import ui
from ..predict import GDMLPredict

try:
    import torch
except ImportError:
    _has_torch = False
else:
    _has_torch = True


CG_STEPS_HIST_LEN = (
    1000 #100  # number of past steps to consider when calculatating solver effectiveness
)
EFF_RESTART_THRESH = -30 #0  # if solver effectiveness is less than that percentage after 'CG_STEPS_HIST_LEN'-steps, a solver restart is triggert (with stronger preconditioner)
EFF_EXTRA_BOOST_THRESH = (
    50  # increase preconditioner more aggressively below this efficiency threshold
)


class CGRestartException(Exception):
    pass


class Iterative(object):
    def __init__(
        self, gdml_train, desc, callback=None, max_processes=None, use_torch=False
    ):

        self.gdml_train = gdml_train
        self.gdml_predict = None
        self.desc = desc

        self.callback = callback

        self._max_processes = max_processes
        self._use_torch = use_torch

        # this will be set once the kernel operator is used on the GPU with pytorch
        self._gpu_batch_size = 0


    #from memory_profiler import profile

    #@profile
    def _init_precon_operator(
        self, task, R_desc, R_d_desc, tril_perms_lin, inducing_pts_idxs, callback=None
    ):

        lam = task['lam']
        lam_inv = 1.0 / lam

        sig = task['sig']

        use_E_cstr = task['use_E_cstr']

        if callback is not None:
            callback = partial(
                callback,
                disp_str='Assembling (partial) kernel matrix',
            )

        K_nm = self.gdml_train._assemble_kernel_mat(
            R_desc,
            R_d_desc,
            tril_perms_lin,
            sig,
            self.desc,
            use_E_cstr=use_E_cstr,
            col_idxs=inducing_pts_idxs,
            callback=callback,
        )

        n, m = K_nm.shape
        K_mm = K_nm[inducing_pts_idxs, :] # yields copy due to non-sequential indexing

        if callback is not None:
            callback = partial(
                callback,
                disp_str='Factorizing',
            )
            callback(NOT_DONE)

        inner = -lam * K_mm + K_nm.T.dot(K_nm)
        L, lower = self._cho_factor_stable(inner, lam)

        b_start, b_size = 0, int(n / 10)  # update in percentage steps of 10
        for b_stop in list(range(b_size, n, b_size)) + [n]:

            K_nm[b_start:b_stop, :] = sp.linalg.solve_triangular(
                L,
                K_nm[b_start:b_stop, :].T,
                lower=lower,
                trans='T',
                overwrite_b=True,
                check_finite=False,
            ).T  # Note: Overwrites K_nm to save memory

            if callback is not None:
                callback(b_stop, n)

            b_start = b_stop
        del L

        L_inv_K_mn = K_nm.T

        if self._use_torch and False:  # TURNED OFF!
            _torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
            L_inv_K_mn_torch = torch.from_numpy(L_inv_K_mn).to(_torch_device)

        global is_primed
        is_primed = False

        def _P_vec(v):

            global is_primed
            if not is_primed:
                is_primed = True
                return v

            if self._use_torch and False:  # TURNED OFF!

                v_torch = torch.from_numpy(v).to(_torch_device)[:, None]
                return (
                    L_inv_K_mn_torch.t().mm(L_inv_K_mn_torch.mm(v_torch)) - v_torch
                ).cpu().numpy() * lam_inv

            else:

                ret = L_inv_K_mn.T.dot(L_inv_K_mn.dot(v))
                ret -= v
                ret *= lam_inv
                return ret

        return sp.sparse.linalg.LinearOperator((n, n), matvec=_P_vec) 

    def _init_kernel_operator(
        self, task, R_desc, R_d_desc, tril_perms_lin, lam, n, callback=None
    ):

        n_train = R_desc.shape[0]

        # dummy alphas
        v_F = np.zeros((n, 1))
        v_E = np.zeros((n_train, 1)) if task['use_E_cstr'] else None

        # Note: The standard deviation is set to 1.0, because we are predicting normalized labels here.
        model = self.gdml_train.create_model(
            task, 'cg', R_desc, R_d_desc, tril_perms_lin, 1.0, v_F, alphas_E=v_E
        )

        self.gdml_predict = GDMLPredict(
            model, max_processes=self._max_processes, use_torch=self._use_torch
        )

        if not self._use_torch:

            if callback is not None:
                callback = partial(callback, disp_str='Optimizing CPU parallelization')
                callback(NOT_DONE)

            self.gdml_predict.prepare_parallel(n_bulk=n_train)

            if callback is not None:
                callback(DONE)

        global is_primed
        is_primed = False

        def _K_vec(v):

            global is_primed
            if not is_primed:
                is_primed = True
                return v

            v_F, v_E = v, None
            if task['use_E_cstr']:
                v_F, v_E = v[:-n_train], v[-n_train:]

            self.gdml_predict.set_alphas(R_d_desc, v_F, alphas_E=v_E)

            if self._use_torch:
                self._gpu_batch_size = self.gdml_predict.get_GPU_batch()

                if self._gpu_batch_size > n_train:
                    self._gpu_batch_size = n_train

            R = task['R_train'].reshape(n_train, -1)
            #print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n')
            #print('------------------------------------------')
            #print('IS FAILLING!!!!!!!!!')
            #print('TYPES', type(R_desc), type(R_d_desc))
            if type(R_d_desc) != list:
                if R_d_desc.shape[-1] != 3:
                    print()
                    print('Inconsistency Problem Befor!!!!')
                    print('R_d_desc TYPE:', type(R_d_desc))
                    print('R_d_desc SHAPE;', R_d_desc.shape)
                    print()
            e_pred, f_pred = self.gdml_predict.predict(R, R_desc, R_d_desc)
            #print('END------------------------------------------')
            #print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n')

            pred = f_pred.ravel()
            if task['use_E_cstr']:
                pred = np.hstack((pred, -e_pred))

            pred -= lam * v
            return pred

        return sp.sparse.linalg.LinearOperator((n, n), matvec=_K_vec)

    
    def _lev_scores(
        self,
        R_desc,
        R_d_desc,
        tril_perms_lin,
        sig,
        lam,
        use_E_cstr,
        n_inducing_pts,
        idxs_ordered_by_lev_score=None,  # importance ordering of columns used to pick the columns to approximate leverage scoresid (optional)
        callback=None,
    ):

        #n_train, dim_d = R_d_desc.shape[:2]
        #dim_i = 3 * int((1 + np.sqrt(8 * dim_d + 1)) / 2)
        n_train, dim_d = R_desc.shape[:]
        dim_i = self.desc.dim_i

        # Convert from training points to actual columns.
        #dim_m = n_inducing_pts * dim_i
        dim_m = np.maximum(1, n_inducing_pts // 4) * dim_i # only use 1/4 of inducing points for leverage score estimate

        # Which columns to use for leverage score approximation?
        if idxs_ordered_by_lev_score is None:
            lev_approx_idxs = np.sort(np.random.choice(n_train*dim_i, dim_m, replace=False)) # random subset of columns
            #lev_approx_idxs = np.s_[
            #    :dim_m
            #]  # first 'dim_m' columns (faster kernel construction)
        else:
            assert len(idxs_ordered_by_lev_score) == n_train * dim_i

            lev_approx_idxs = np.sort(
                idxs_ordered_by_lev_score[-dim_m:]
            )  # choose 'dim_m' columns according to provided importance ordering

        if callback is not None:
            callback = partial(
                callback,
                disp_str='Approx. leverage scores (1/3 assembling matrix)',
            )

        K_nm = self.gdml_train._assemble_kernel_mat(
            R_desc,
            R_d_desc,
            tril_perms_lin,
            sig,
            self.desc,
            use_E_cstr=use_E_cstr,
            col_idxs=lev_approx_idxs,
            callback=callback,
        )
        K_mm = K_nm[lev_approx_idxs, :] # yields copy due to non-sequential indexing

        if callback is not None:
            callback = partial(
                callback, disp_str='Approx. leverage scores (2/3 factoring)'
            )
            callback(NOT_DONE)

        L, lower = self._cho_factor_stable(-K_mm)
        del K_mm

        callback(DONE)

        if callback is not None:
            callback = partial(
                callback, disp_str='Approx. leverage scores (3/3 constructing)'
            )

        n = K_nm.shape[0]
        b_start, b_size = 0, int(n / 10)  # update in percentage steps of 10
        for b_stop in list(range(b_size, n, b_size)) + [n]:

            K_nm[b_start:b_stop, :] = sp.linalg.solve_triangular(
                L,
                K_nm[b_start:b_stop, :].T,
                lower=lower,
                trans='T',
                overwrite_b=True,
                check_finite=False,
            ).T  # Note: Overwrites K_nm to save memory

            if callback is not None:
                callback(b_stop, n)

            b_start = b_stop
        B = K_nm.T
        del L

        B_BT_lam = B.dot(B.T)
        B_BT_lam[np.diag_indices_from(B_BT_lam)] += lam

        # Leverage scores for all columns.
        # lev_scores = np.einsum('ij,ij->j', B, np.linalg.solve(B_BT_lam, B))

        # Leverage scores for all columns.
        #C, C_lower = sp.linalg.cho_factor(
        #    B_BT_lam, overwrite_a=True, check_finite=False
        #)
        C, C_lower = self._cho_factor_stable(B_BT_lam)
        del B_BT_lam

        B = sp.linalg.solve_triangular(
            C, B, lower=C_lower, trans='T', overwrite_b=True, check_finite=False
        )
        C_B = B
        lev_scores = np.einsum('i...,i...->...', C_B, C_B)

        return np.argsort(lev_scores)

        # # Try to group columns by molecule to speed up kernel matrix generation:
        # lev_scores = np.around(
        #     lev_scores, decimals=2
        # )  # Round leverage scoresid to second decimal place, then sort by score and training point index combined.

        # point_idxs = np.tile(np.arange(n_train)[:, None], (1, dim_i)).ravel()
        # idxs_ordered_by_lev_score = np.lexsort(
        #     (point_idxs, lev_scores)
        # )  # sort by 'lev_scores' then by 'point_idxs'

        # return idxs_ordered_by_lev_score

    # performs a cholesky decompostion of a matrix, but regularizes the matrix (if neeeded) until its positive definite
    def _cho_factor_stable(self, M, min_eig=None):
        """
        Performs a Cholesky decompostion of a matrix, but regularizes
        as needed until its positive definite.

        Parameters
        ----------
            M : :obj:`numpy.ndarray`
                Matrix to factorize.
            min_eig : float
                Force lowest eigenvalue to
                be a certain (positive) value
                (default: machine precision)

        Returns
        -------
            :obj:`numpy.ndarray`
                Matrix whose upper or lower triangle contains the Cholesky factor of a. Other parts of the matrix contain random data.
            boolean
                Flag indicating whether the factor is in the lower or upper triangle
        """

        eps = np.finfo(float).eps
        eps_mag = int(np.floor(np.log10(eps)))

        if min_eig is None:
            min_eig = eps
        else:
            assert min_eig > 0

        lo_eig = sp.linalg.eigh(M, eigvals_only=True, eigvals=(0, 0))
        if lo_eig < min_eig:
            sgn = 1 if lo_eig <= 0 else -1
            M[np.diag_indices_from(M)] += sgn * (min_eig - lo_eig)

        for reg in 10.0 ** np.arange(
            eps_mag, 2
        ):  # regularize more and more aggressively (strongest regularization: 1)
            try:

                L, lower = sp.linalg.cho_factor(
                    M, overwrite_a=False, check_finite=False
                )

            except np.linalg.LinAlgError as e:

                if 'not positive definite' in str(e):
                    M[np.diag_indices_from(M)] += reg
                else:
                    raise e
            else:
                return L, lower

    def solve(
        self,
        task,
        R_desc,
        R_d_desc,
        tril_perms_lin,
        y,
        y_std,
        save_progr_callback=None,
    ):

        global num_iters, start, resid, avg_tt, m

        n_train, n_atoms = task['R_train'].shape[:2]
        dim_i = 3 * n_atoms

        sig = task['sig']
        lam = task['lam']

        # these keys are only present if the task was created from an existing model
        alphas0_F = task['alphas0_F'] if 'alphas0_F' in task else None
        alphas0_E = task['alphas0_E'] if 'alphas0_E' in task else None
        num_iters0 = task['solver_iters'] if 'solver_iters' in task else 0

        n_inducing_pts_init = task['n_inducing_pts_init'].copy()
        if 'inducing_pts_idxs' in task: # only available if task was created from an existing model
            n_inducing_pts_init = len(task['inducing_pts_idxs']) // (3*n_atoms)

        # How many inducing points to use (for Nystrom approximation, as well as the approximation of leverage scores).
        # Note: this number is automatically increased if necessary.
        n_inducing_pts = min(n_train, n_inducing_pts_init)

        if self.callback is not None:
            self.callback = partial(
                self.callback,
                disp_str='Constructing preconditioner',
            )
        subtask_callback = partial(ui.sec_callback, main_callback=self.callback)

        idxs_ordered_by_lev_score = None
        if 'inducing_pts_idxs' in task:
            inducing_pts_idxs = task['inducing_pts_idxs']
        else:
            # Determine good inducing points.
            idxs_ordered_by_lev_score = self._lev_scores(
                R_desc,
                R_d_desc,
                tril_perms_lin,
                sig,
                lam,
                False,  # use_E_cstr
                n_inducing_pts,
                callback=subtask_callback,
            )
            dim_m = n_inducing_pts*dim_i
            inducing_pts_idxs = np.sort(idxs_ordered_by_lev_score[-dim_m:])

        start = timeit.default_timer()
        P_op = self._init_precon_operator(
            task,
            R_desc,
            R_d_desc,
            tril_perms_lin,
            inducing_pts_idxs,
            callback=subtask_callback,
        )
        stop = timeit.default_timer()

        if self.callback is not None:
            dur_s = stop - start
            sec_disp_str = 'took {:.1f} s'.format(dur_s) if dur_s >= 0.1 else ''
            self.callback(DONE, sec_disp_str=sec_disp_str)

        if self.callback is not None:
            self.callback = partial(
                self.callback,
                disp_str='Initializing solver',
            )
        subtask_callback = partial(ui.sec_callback, main_callback=self.callback)

        n = P_op.shape[0]
        K_op = self._init_kernel_operator(
            task, R_desc, R_d_desc, tril_perms_lin, lam, n, callback=subtask_callback
        )

        num_iters = int(num_iters0)

        if task['use_E_cstr'] and self._use_torch:
            print('NOT IMPLEMENTED!!!')
            sys.exit()

        if self.callback is not None:

            num_devices = (
                mp.cpu_count() if self._max_processes is None else self._max_processes
            )
            if self._use_torch:
                num_devices = (
                    torch.cuda.device_count()
                    if torch.cuda.is_available()
                    else torch.get_num_threads()
                )
            hardware_str = '{:d} {}{}{}'.format(
                num_devices,
                'GPU' if self._use_torch and torch.cuda.is_available() else 'CPU',
                's' if num_devices > 1 else '',
                '[PyTorch]' if self._use_torch else '',
            )

            self.callback(NOT_DONE, sec_disp_str=None)

        start = 0
        resid = 0
        avg_tt = 0

        global alpha_t, eff, steps_hist, callback_disp_str

        alpha_t = None
        steps_hist = collections.deque(
            maxlen=CG_STEPS_HIST_LEN
        )  # moving average window for step history

        increase_ip = False

        callback_disp_str = 'Initializing solver'
        def _cg_status(xk):
            global num_iters, start, resid, alpha_t, avg_tt, m, eff, steps_hist, callback_disp_str

            stop = timeit.default_timer()
            tt = 0.0 if start == 0 else (stop - start)
            avg_tt += tt
            start = timeit.default_timer()

            old_resid = resid
            resid = inspect.currentframe().f_back.f_locals['resid']

            step = 0 if num_iters == num_iters0 else resid - old_resid
            steps_hist.append(step)

            steps_hist_arr = np.array(steps_hist)
            steps_hist_all = np.abs(steps_hist_arr).sum()
            steps_hist_ratio = (
                (-steps_hist_arr.clip(max=0).sum() / steps_hist_all)
                if steps_hist_all > 0
                else 1
            )
            eff = 0 if num_iters == num_iters0 else (int(100 * steps_hist_ratio) - 50) * 2

            if tt > 0.0 and num_iters % int(np.ceil(1.0 / tt)) == 0:  # once per second

                train_rmse = resid / np.sqrt(len(y))

                if self.callback is not None:

                    callback_disp_str = 'Training error (RMSE): forces {:.4f}'.format(train_rmse)
                    self.callback(
                        NOT_DONE,
                        disp_str=callback_disp_str,
                        sec_disp_str=(
                            '{:d} iter @ {} iter/s [eff: {:d}%] k: {:d}'.format(
                                num_iters,
                                '{:.1f}'.format(1.0 / tt),
                                eff,
                                n_inducing_pts,
                            )
                        ),
                    )

            # Write out current solution as a model file once every 2 minutes (give or take).
            if tt > 0.0 and num_iters % int(np.ceil(2 * 60.0 / tt)) == 0:

                # TODO: support for +E constraints (done?)
                alphas_F, alphas_E = -xk, None
                if task['use_E_cstr']:
                    alphas_F, alphas_E = -xk[:-n_train], -xk[-n_train:]

                unconv_model = self.gdml_train.create_model(
                    task,
                    'cg',
                    R_desc,
                    R_d_desc,
                    tril_perms_lin,
                    y_std,
                    alphas_F,
                    alphas_E=alphas_E,
                    solver_resid=resid,
                    solver_iters=num_iters + 1,
                    norm_y_train=np.linalg.norm(y),
                    inducing_pts_idxs=inducing_pts_idxs,
                )

                # recover integration constant
                n_train = task['E_train'].shape[0]
                R = task['R_train'].reshape(n_train, -1)

                self.gdml_predict.set_alphas(R_d_desc, alphas_F, alphas_E=alphas_E)
                E_pred, _ = self.gdml_predict.predict(R)
                E_pred *= y_std
                E_ref = np.squeeze(task['E_train'])

                unconv_model['c'] = np.sum(E_ref - E_pred) / E_ref.shape[0]

                if save_progr_callback is not None:
                    save_progr_callback(unconv_model)

            num_iters += 1

            n_train = task['E_train'].shape[0]
            if (
                len(steps_hist) == CG_STEPS_HIST_LEN
                and eff <= EFF_RESTART_THRESH
                and n_inducing_pts < n_train
            ):

                alpha_t = xk
                raise CGRestartException

        alphas0 = None
        #alphas0 = np.random.uniform(low=-1, high=1, size=y.shape)
        if alphas0_F is not None:  # TODO: improve me: this iwll not workt with E_cstr
            alphas0 = -alphas0_F

        if alphas0_E is not None:
            alphas0_E *= -1  # TODO: is this correct (sign)?
            alphas0 = np.hstack((alphas0, alphas0_E))

        num_restarts = 0
        while True:
            try:
                alphas, info = sp.sparse.linalg.cg(
                    -K_op,
                    y,
                    x0=alphas0 if alpha_t is None else alpha_t,
                    M=P_op,
                    tol=task['solver_tol'], # norm(residual) <= max(tol*norm(b), atol)
                    atol=None,
                    maxiter=3 * n_atoms * n_train * 10, # allow 10x as many iterations as theoretically needed (at perfect precision)
                    callback=_cg_status,
                )
                alphas = -alphas

            except CGRestartException:

                num_restarts += 1
                steps_hist.clear()

                n_inducing_pts += (
                    5 if eff <= EFF_EXTRA_BOOST_THRESH else 1
                )  # increase more agressively if convergence is especially weak
                n_inducing_pts = min(n_inducing_pts, n_train)

                subtask_callback = partial(ui.sec_callback, main_callback=partial(self.callback, disp_str=callback_disp_str))

                if (
                    num_restarts == 1 or num_restarts % 10 == 0 or idxs_ordered_by_lev_score is None
                ):  # recompute leverate scoresid on first restart (first approximation is bad) and every 10 restarts.

                    # Use leverage scoresid from last run to estimate better ones this time.
                    idxs_ordered_by_lev_score = self._lev_scores(
                        R_desc,
                        R_d_desc,
                        tril_perms_lin,
                        sig,
                        lam,
                        False,  # use_E_cstr
                        n_inducing_pts,
                        idxs_ordered_by_lev_score=idxs_ordered_by_lev_score,
                        callback=subtask_callback,
                    )

                dim_m = n_inducing_pts*dim_i
                inducing_pts_idxs = np.sort(
                    idxs_ordered_by_lev_score[-dim_m:]
                )

                del P_op
                P_op = self._init_precon_operator(
                    task,
                    R_desc,
                    R_d_desc,
                    tril_perms_lin,
                    inducing_pts_idxs,
                    callback=subtask_callback,
                )

            else:
                break

        is_conv = info == 0

        if self.callback is not None:

            is_conv_warn_str = '' if is_conv else ' (NOT CONVERGED)'
            self.callback(
                DONE,
                disp_str='Training on {:,} points{}'.format(n_train, is_conv_warn_str),
                sec_disp_str=(
                    '{:d} iter @ {} iter/s'.format(
                        num_iters,
                        '{:.1f}'.format(num_iters / avg_tt) if avg_tt > 0 else '--',
                    )
                ),
                done_with_warning=not is_conv,
            )

        train_rmse = resid / np.sqrt(len(y))
        return alphas, num_iters, resid, train_rmse, inducing_pts_idxs, is_conv
