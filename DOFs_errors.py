import numpy as np
import sys
from tqdm import tqdm
import scipy as sp
from scipy import stats
import os
import glob

DOFs_file = sys.argv[1] #File(s) obtained from the DOFs_predictions.py script. An npz file or a directory (if many npz files)

if DOFs_file[-1] == '/':
    print('Working with dir...')
    print('All npz files in the given dirs will be considered for each model')
    dirs=True
else:
    dirs=False

if dirs:
   FILES = sorted(glob.glob(DOFs_file+'*DOFs*.npz'))
   for kk in range(len(FILES)):
       if 'checkpoint.npz' in FILES[kk]:
           FILES.remove(FILES[kk])
   if len(FILES) > 1:
       FILES = FILES[1:] + [FILES[0]]
   print('Check if the order of files is consistent...')
   E_DOFs_temp = []
   F_DOFs_temp = []
   for i in range(len(FILES)):
       print(FILES[i])
       DOFs_data = np.load(FILES[i], allow_pickle=True)
       if i == 0:
           E_base = DOFs_data['E_base']
           F_base = DOFs_data['F_base']
       E_DOFs_temp.append(DOFs_data['E_DOFs'])
       F_DOFs_temp.append(DOFs_data['F_DOFs'])
       del DOFs_data       

   E_DOFs = np.concatenate(E_DOFs_temp)
   del E_DOFs_temp
   F_DOFs = np.concatenate(F_DOFs_temp)
   del F_DOFs_temp
   print()

else:

    DOFs_data = np.load(DOFs_file, allow_pickle=True)


    E_base = DOFs_data['E_base']
    F_base = DOFs_data['F_base']

    E_DOFs = DOFs_data['E_DOFs']
    F_DOFs = DOFs_data['F_DOFs']

    if 'DOF_order' in DOFs_data.files:
        DOF_order = DOFs_data['DOF_order']
        print('There is a random order...')
        order = True
    else:
        order = False
    del DOFs_data

print('Shape of Base E an F:', E_base.shape, F_base.shape)
print('Shape of DOFs E an F:', E_DOFs.shape, F_DOFs.shape) 

E_RMSEs = []
F_RMSEs = []
for i in tqdm(range(E_DOFs.shape[0])):
        e_error = np.sqrt(np.mean((E_base.ravel() - E_DOFs[i,:,:].ravel()) ** 2))
        f_error = np.sqrt(np.mean((F_base.ravel() - F_DOFs[i,:,:].ravel()) ** 2))
        E_RMSEs.append(e_error)
        F_RMSEs.append(f_error)

E_RMSEs = np.array(E_RMSEs)
F_RMSEs = np.array(F_RMSEs)

print('min max values:')
print('Energy:', np.amin(E_RMSEs), np.amax(E_RMSEs), np.where(E_RMSEs == np.amin(E_RMSEs))[0], np.where(E_RMSEs == np.amax(E_RMSEs))[0])

print('Forces:', np.amin(F_RMSEs), np.amax(F_RMSEs), np.where(F_RMSEs == np.amin(F_RMSEs))[0], np.where(F_RMSEs == np.amax(F_RMSEs))[0])
 


n_atoms = F_DOFs.shape[-1] / 3

print('Total Atoms:',n_atoms)
print()

std = np.std(F_RMSEs)
print('Standard deviation of Forces RMSEs:', std)
mean = np.mean(F_RMSEs)
print('Mean of Forces RMSEs:', mean)
median= np.median(F_RMSEs)
print('Median of Forces RMSEs:', median)
skew = sp.stats.skew(F_RMSEs)
print('Skewness of Forces RMSEs:', skew)
kurtosis = sp.stats.kurtosis(F_RMSEs)
print('Kurtosis of Forces RMSEs:', kurtosis)

Save=True
Q = [10, 20, 30, 40, 50, 60, 70, 80, 90]
print('Percentiles:')
for q in Q:
    percentile = np.percentile(F_RMSEs, q)
    print(q, percentile)
    out_DOFs = np.where(F_RMSEs < percentile)[0]
    print('out DOFs:', out_DOFs.shape, out_DOFs)
    print()
    if Save:
        dir_name = "Q"+str(q)
        if not os.path.exists(dir_name):
           os.mkdir(dir_name)
        if dirs:
            prefix = FILES[-1].split('/')[-1].split('.')[0]
        else:
            prefix = DOFs_file.split('/')[-1].split('.')[0]
        np.savez(dir_name+'/'+prefix+'_'+dir_name+'_idxs.npz', idxs = out_DOFs)

