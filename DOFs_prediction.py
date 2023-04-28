import numpy as np

from ase.io.trajectory import Trajectory
from ase.io import read
from sgdml.predict import GDMLPredict, share_array
from sgdml.utils import desc
import sys
from tqdm import tqdm
import random


def get_random_idxs(N_samples, n_mol):

    idxs = []
    for i in range(N_samples):
        if i > 0:
            while n in idxs:
                n = random.randint(0,n_mol-1)
            else:
                idxs.append(n)
        else:
            n = random.randint(0,n_mol-1)
            idxs.append(n)

    return np.array(sorted(idxs))

def init_model(model, DOF = None, r_d_desc_train = None): #r_desc=None, r_d_desc=None):

    sig = model['sig']
    n_train = model['R_desc'].shape[1]
    n_perms = model['perms'].shape[0]
    tril_perms_lin = model['tril_perms_lin']


    R_desc = model['R_desc']
    R_d_desc_alpha = model['R_d_desc_alpha']
    if DOF is None:
        print('Shape of R_desc in model:', R_desc.shape)
        print()
        R_d_desc_alpha = model['R_d_desc_alpha']
        print('Shape of R_d_desc_alpha in model:', R_d_desc_alpha.shape)
        print()

    else:
        R_desc[DOF, :] = 0
        R_d_desc_alpha[:,DOF] = 0 

    R_desc_perms = (
        np.tile(R_desc.T, n_perms)[:, tril_perms_lin]
        .reshape(n_train, n_perms, -1, order='F')
        .reshape(n_train * n_perms, -1)
    )
    R_desc_perms, R_desc_perms_shape = share_array(R_desc_perms)


    R_d_desc_alpha_perms = (
        np.tile(R_d_desc_alpha, n_perms)[:, tril_perms_lin]
        .reshape(n_train, n_perms, -1, order='F')
        .reshape(n_train * n_perms, -1)
    )
    R_d_desc_alpha_perms, R_d_desc_alpha_perms_shape = share_array(R_d_desc_alpha_perms)

    return R_desc_perms, R_desc_perms_shape, R_d_desc_alpha_perms, R_d_desc_alpha_perms_shape

def set_alphas(R_d_desc, alphas): #, tril_perms_lin):  # TODO: document me, this only sets alphas_F


    r_dim = R_d_desc.shape[2]
    n_train = R_d_desc.shape[0]
    R_d_desc_alpha = np.einsum(
        'kji,ki->kj', R_d_desc, alphas.reshape(-1, r_dim)
    )

    return R_d_desc_alpha


def _predict_wkr(wkr_start_stop, chunk_size, r_desc, sig, n_perms, R_desc_perms, R_desc_perms_shape, R_d_desc_alpha_perms, R_d_desc_alpha_perms_shape):

    wkr_start, wkr_stop = wkr_start_stop

    R_desc_perms = np.frombuffer(R_desc_perms).reshape(R_desc_perms_shape)

    R_d_desc_alpha_perms = np.frombuffer(R_d_desc_alpha_perms).reshape(R_d_desc_alpha_perms_shape)


    dim_d = r_desc.shape[0]
    dim_c = chunk_size * n_perms
    
    # pre-allocation

    diff_ab_perms = np.empty((dim_c, dim_d))
    a_x2 = np.empty((dim_c,))
    mat52_base = np.empty((dim_c,))

    mat52_base_fact = 5.0 / (3 * sig ** 3)
    diag_scale_fact = 5.0 / sig
    sqrt5 = np.sqrt(5.0)

    E_F = np.zeros((dim_d + 1,))
    F = E_F[1:]

    wkr_start *= n_perms
    wkr_stop *= n_perms

    b_start = wkr_start
    for b_stop in list(range(wkr_start + dim_c, wkr_stop, dim_c)) + [wkr_stop]:

        rj_desc_perms = R_desc_perms[b_start:b_stop, :]
        rj_d_desc_alpha_perms = R_d_desc_alpha_perms[b_start:b_stop, :]

        # Resize pre-allocated memory for last iteration, if chunk_size is not a divisor of the training set size.
        # Note: It's faster to process equally sized chunks.
        c_size = b_stop - b_start
        if c_size < dim_c:
            diff_ab_perms = diff_ab_perms[:c_size, :]
            a_x2 = a_x2[:c_size]
            mat52_base = mat52_base[:c_size]

        np.subtract(
            np.broadcast_to(r_desc, rj_desc_perms.shape),
            rj_desc_perms,
            out=diff_ab_perms,
        )
        norm_ab_perms = sqrt5 * np.linalg.norm(diff_ab_perms, axis=1)

        np.exp(-norm_ab_perms / sig, out=mat52_base)
        mat52_base *= mat52_base_fact
        np.einsum(
            'ji,ji->j', diff_ab_perms, rj_d_desc_alpha_perms, out=a_x2
        )  # colum wise dot product

        F += (a_x2 * mat52_base).dot(diff_ab_perms) * diag_scale_fact
        mat52_base *= norm_ab_perms + sig

        F -= mat52_base.dot(rj_d_desc_alpha_perms)
        E_F[0] += a_x2.dot(mat52_base)

        b_start = b_stop

    return E_F

def predict(r, lat_and_inv, b_arr, a_arr, d_arr, desc_type, std, n_train, c, sig, n_perms, R_desc_perms, R_desc_perms_shape, R_d_desc_alpha_perms, R_d_desc_alpha_perms_shape, DOF = None):

    n_atoms = r.shape[0]
    r_desc, r_d_desc = desc.from_r(r, lat_and_inv, b_arr=b_arr, a_arr=a_arr, d_arr=d_arr, desc_type=desc_type)

    if DOF is not None:
        r_desc[DOF] = 0
        r_d_desc[DOF,:] = 0

    res = _predict_wkr((0, n_train), n_train, r_desc,sig, n_perms, R_desc_perms, R_desc_perms_shape, R_d_desc_alpha_perms, R_d_desc_alpha_perms_shape)
    
    res *= std

    E = res[0].reshape(-1) + c
    F = res[1:].reshape(1, -1).dot(r_d_desc)
    return E, F

Model_name = sys.argv[1] #Default GDML model
Dataset_name = sys.argv[2] #Dataset from which the ML model was trained

Model = np.load(Model_name, allow_pickle=True)
Dataset = np.load(Dataset_name, allow_pickle=True)


R_train = Dataset['R'][Model['idxs_train'], :, :]

other_idxs = np.delete(np.arange(Dataset['R'].shape[0]), np.concatenate((Model['idxs_train'], Model['idxs_valid'])))

N_samples = 3000 #sys.argv[3]
n_conf = other_idxs.shape[0]

sample_idxs = get_random_idxs(N_samples, n_conf)
R_base = Dataset['R'][other_idxs[sample_idxs], :, :]


sig = Model['sig']
n_perms = Model['perms'].shape[0]

lat_and_inv = None

if 'desc_type' not in Model.files:
   desc_type = 1
else:
   desc_type = Model['desc_type']


std = Model['std'] if 'std' in Model.files else 1.0
n_train = Model['idxs_train'].shape[0]
c = Model['c']

print('Computing R_d_desc of training set...')
print()

r_d_desc_train = []
for i in range(R_train.shape[0]):
    r = R_train[i,:,:]
    _, r_d_desc = desc.from_r(r, lat_and_inv, b_arr, a_arr, d_arr, desc_type)
    r_d_desc_train.append(r_d_desc)

r_d_desc_train = np.array(r_d_desc_train)

R_desc_perms, R_desc_perms_shape, R_d_desc_alpha_perms, R_d_desc_alpha_perms_shape = init_model(Model, DOF = None, r_d_desc_train = r_d_desc_train)

print('Shape of r_d_desc_train:', r_d_desc_train.shape)
print()

E_base = []
F_base = []

print('Computing the base predictions...')
print()

for i in range(R_base.shape[0]):

    e, f = predict(R_base[i,:,:], lat_and_inv, b_arr, a_arr, d_arr, desc_type, std, n_train, c,sig, n_perms, R_desc_perms, R_desc_perms_shape, R_d_desc_alpha_perms, R_d_desc_alpha_perms_shape)

    E_base.append(e)
    F_base.append(f.ravel())

E_base = np.array(E_base)
F_base = np.array(F_base)

print('Shape of E and F of base prediction:', E_base.shape, F_base.shape)
print()

dim_d  = Model['R_desc'].shape[0]

print('Computing predictions masking certain DOFs...')
print()
E_DOFs = []
F_DOFs = []
count = 1
for DOF in range(dim_d):
    R_desc_perms, R_desc_perms_shape, R_d_desc_alpha_perms, R_d_desc_alpha_perms_shape = init_model(Model, DOF=DOF, r_d_desc_train = r_d_desc_train)
    E = []
    F = []
    for i in range(R_base.shape[0]):
        e, f = predict(R_base[i,:,:], lat_and_inv, b_arr, a_arr, d_arr, desc_type, std, n_train, c, sig, n_perms, R_desc_perms, R_desc_perms_shape, R_d_desc_alpha_perms, R_d_desc_alpha_perms_shape, DOF=DOF)
        
        E.append(e)
        F.append(f.ravel())

    E_DOFs.append(np.array(E))
    F_DOFs.append(np.array(F))
    count += 1
    if count == 30:
       np.savez(Dataset_name.split('/')[-1].split('.')[0]+'_'+Model_name.split('/')[-1].split('train')[-1].split('-')[0]+'_pred_DOFs_checkpoint.npz', E_base = E_base, F_base = F_base, E_DOFs = np.array(E_DOFs), F_DOFs = np.array(F_DOFs)) 
       count = 1

E_DOFs = np.array(E_DOFs)
F_DOFs = np.array(F_DOFs)

print('Shape of E and F of DOFs predictions:', E_DOFs.shape, F_DOFs.shape)
print()

np.savez(Dataset_name.split('/')[-1].split('.')[0]+'_'+Model_name.split('/')[-1].split('train')[-1].split('-')[0]+'_pred_DOFs.npz', E_base = E_base, F_base = F_base, E_DOFs = E_DOFs, F_DOFs = F_DOFs)

