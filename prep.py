import h5py
import sigpy as sp
import numpy as np
import glob, os
from sigpy.mri.app import EspiritCalib

def prepKspace(kspace):
    # in kspace: (nSL, nCH, nRO, NPE)

    # out kspace: (nSL, nCH, nRO, nPE)
    # out sens: (nSL, nCH, nRO, nPE)
    # out img: (nSL, nRO, nPE)

    sens = kspace * 0

    for sl in range(len(kspace)):
        emaps = EspiritCalib(kspace[sl]).run()
        sens[sl] = emaps.squeeze()
    return kspace, sens

basedir = '/scratch/users/kanghyun/MTLData/ankle_elbow'

file_list = glob.glob(os.path.join(basedir, '*/*.h5'))


for file in file_list:
    with h5py.File(file, 'r') as hr:
        kspace = hr['kspace'][:]

    nsl, nch, nx, ny = kspace.shape
    if ny < nx: 
        kspace = sp.resize(kspace, (nsl,nch,nx,nx*2))
    else:
        kspace = sp.resize(kspace, (nsl,nch,nx,nx))
    kspace, sens = prepKspace(kspace)
    print(file, kspace.shape, sens.shape)

    with h5py.File(file, 'w') as hw:
        hw.create_dataset('kspace', data=np.complex64(kspace))
        hw.create_dataset('sens', data=np.complex64(sens))