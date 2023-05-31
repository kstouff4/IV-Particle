import sys
from sys import path as sys_path
sys_path.append('../')
from varap.io.load_Allen3DMERFISH import Allen3DMerfishLoader
from varap.loss.particles import ParticleLoss_full, ParticleLoss_restricted
from varap.optim.band import optimize
from varap.io.writeOut import writeParticleVTK

import numpy as np
import torch

dtype = torch.cuda.FloatTensor
torch.set_default_tensor_type(dtype)

# fpath = '/content/drive/My Drive/Kaitlin/'
fpath = '/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenMerfish/XnuX_Aligned/top20MI/'
opath = '/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenMerfish/ZnuX_Aligned/top20MI/sig0.025/'

bw = 20
sig = .025

a = Allen3DMerfishLoader(fpath,[0,0,0.100])
particles,features = a.getSizes()
a.subSampleStratified(opath,sig,alpha=0.75)
a.saveToPKL(opath + 'subsampledObject.pkl')

## First step ##

# Define loss functions
numFiles = a.getNumberOfFiles()

for f in range(numFiles):
    HZ,Hnu_Z,LZ,Lnu_Z = a.getHighLowPair(f)
    outfile = a.getFilename_subsample(f).split('/')[-1].replace('.npz','_optimal')
    
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
        np.savez_compressed(opath + outfile, Z=nZ, nu_Z=nnu_Z)
        return nZ, nnu_Z

    Z, nu_Z = optimize(L_restricted.loss, x_init, dxmax, nb_iter=10, callback=callback_restricted)

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
    writeParticleVTK(fpath+outfile+'.npz')
