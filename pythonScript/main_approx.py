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

HZ, Hnu_Z, LZ, Lnu_Z = load(fpath, Data)


bw = 75
sig = .4

L = ParticleLoss_restricted(sig, HZ, Hnu_Z, LZ, bw=bw)
Z, nu_Z = optimize(LZ, Lnu_Z, L.loss, sig, flag='restricted')

L = ParticleLoss_full(sig, HZ, Hnu_Z, bw=bw)
optimize(Z, nu_Z, L.loss, sig, flag='all')
