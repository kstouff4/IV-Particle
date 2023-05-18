from varap.io.load_allen import load
from varap.loss.particles import ParticleLoss_full, ParticleLoss_restricted
from varap.optim.band import optimize

import numpy as np
import torch

dtype = torch.cuda.FloatTensor
torch.set_default_tensor_type(dtype)

# fpath = '/content/drive/My Drive/Kaitlin/'
fpath = '../data/'
Data = 'Allen'

outfile = 'out'

HZ, Hnu_Z, LZ, Lnu_Z = load(fpath, Data)


bw = 75
sig = .4

## First step ##

# Define loss functions
L_restricted = ParticleLoss_restricted(sig, HZ, Hnu_Z, LZ, bw=bw)

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
