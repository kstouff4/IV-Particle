from sys import path as sys_path

sys_path.append('../')

from varap.io.load_allen import load, loadFromPath
from varap.loss.particles import ParticleLoss_full, ParticleLoss_restricted, ParticleLoss_ranges
from varap.optim.band import optimize
from varap.io.writeOut import writeParticleVTK

import numpy as np
import torch

dtype = torch.cuda.FloatTensor
torch.set_default_tensor_type(dtype)

# fpath = '/content/drive/My Drive/Kaitlin/'
fpath = '../data/'
fpathH = fpath + 'AllenAtlas_ZnuZ_sig0.1.npz'
fpathL = fpath + 'AllenAtlas_ZnuZinit_sig0.2_originalZnu_ZwC2.37037037037037_sig0.2_uniform.npz'

outfile = 'AllenAtlas_out_sig0.2'

HZ, Hnu_Z, LZ, Lnu_Z = loadFromPath(fpathH, fpathL)

HZ = torch.from_numpy(HZ).type(dtype)
Hnu_Z = torch.from_numpy(Hnu_Z).type(dtype)
LZ = torch.from_numpy(LZ).type(dtype)
Lnu_Z = torch.from_numpy(Lnu_Z).type(dtype)


bw = 75
sig = .2

ranges = ParticleLoss_ranges(sig, HZ, Hnu_Z, LZ, Lnu_Z)

## First step ##

# Define loss functions

L_restricted = ParticleLoss_restricted(sig, HZ, Hnu_Z, LZ, bw=bw, ranges=ranges)

# Define initial values
x_init = [torch.tensor(Lnu_Z).type(dtype).sqrt()]
print("x_init ", x_init)
print(x_init[0].type())

dxmax = [x_init[0].mean()]
print(dxmax[0].type())


def callback_restricted(xu):
    nZ = LZ
    nnu_Z = xu[0].detach().cpu().numpy() ** 2
    np.savez_compressed(fpath + outfile, Z=nZ, nu_Z=nnu_Z)
    return nZ, nnu_Z


Z, nu_Z = optimize(L_restricted.loss, x_init, dxmax, nb_iter=20, callback=callback_restricted)

## Second step ##

# Define loss functions
L_all = ParticleLoss_full(sig, HZ, Hnu_Z, bw=bw)

# Define initial values
x_init = [torch.tensor(Z).type(dtype), torch.tensor(nu_Z).type(dtype).sqrt()]
print("x_init ", x_init)

dxmax = [sig, x_init[1].mean()]
print(dxmax)


def callback_all(xu):
    nZ = xu[0].detach().cpu().numpy()
    nnu_Z = xu[1].detach().cpu().numpy() ** 2
    np.savez_compressed(fpath + outfile, Z=nZ, nu_Z=nnu_Z)
    return nZ, nnu_Z


optimize(L_all.loss, x_init, dxmax, nb_iter=20, callback=callback_all)
writeParticleVTK(fpath + outfile + '.npz')
