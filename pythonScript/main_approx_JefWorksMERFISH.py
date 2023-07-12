import os
import sys
from sys import path as sys_path
sys_path.append('../')

from varap.io.load_JEFWorksMERFISHcellTypes import JEFWorksLoader
from varap.loss.particles import ParticleLoss_full, ParticleLoss_restricted, ParticleLoss_ranges
from varap.optim.band import optimize
from varap.io.writeOut import writeParticleVTK

import numpy as np
import torch 

dtype = torch.cuda.FloatTensor
torch.set_default_tensor_type(dtype)

# fpath = '/content/drive/My Drive/Kaitlin/'
fpathO = '/cis/home/kstouff4/Documents/MeshRegistration/Particles/FanMERFISH/'
fpath = '/cis/home/kstouff4/Documents/MeshRegistration/Particles/FanMERFISH/initialHighLowAllCells.pkl'
opath = '/cis/home/kstouff4/Documents/MeshRegistration/Particles/FanMERFISH/sig0.2/'

if (not os.path.exists(opath)):
    os.mkdir(opath) 

bw = 10
sig = 0.2

genes = ['Astro1', 'Astro2','Astro3','Astro4','Astro5','Cort_Excit_Neu1','Cort_Excit_Neu2','Endothelial1','Endothelial2','Ependymal','Excit_Granule','Excit_Neu1','Excit_Neu2','Excit_Pyr_Neu1','Excit_Pyr_Neu2','GABA_Inter1','GABA_Inter2','GABA_Estro_Recept_Neu','Inhib_Inter','Microglia1','Microglia2','OL_Neu','OL_Prog1','OL_Prog2','OL1','OL2','OL3','OL4','OL5','OL6','Pericyte']

a = JEFWorksLoader(fpath,[0.0,0.0,0.0],featNames=genes,dimEff=2,numF=31)

## First step ##

# Define loss functions
numFiles = a.getNumberOfFiles()
print("number of files optimizing: ", numFiles)

for f in range(numFiles):
    nHZ,nHnu_Z,nLZ,nLnu_Z = a.getHighLowPair(f)
    HZ = torch.tensor(nHZ).type(dtype)
    Hnu_Z = torch.tensor(nHnu_Z).type(dtype)
    LZ = torch.tensor(nLZ).type(dtype)
    Lnu_Z = torch.tensor(nLnu_Z).type(dtype)
    
    outfile = a.getFilename_subsample(f).split('/')[-1].replace('.npz','_optimal')
    ranges = ParticleLoss_ranges(sig, HZ, Hnu_Z, LZ, Lnu_Z)
    print(torch.allclose(LZ,ranges.Z))
    
    # reset variables to sorted versions
    HZ = ranges.X
    Hnu_Z = ranges.nu_X
    LZ = ranges.Z
    Lnu_Z = ranges.nu_Z

    #np.savez_compressed(opath + outfile + '_initialSort',Z=LZ.detach().cpu().numpy(),nu_Z=Lnu_Z.detach().cpu().numpy())
    #writeParticleVTK(opath+outfile+'_initialSort.npz',featNames=genes)
    
    L_restricted = ParticleLoss_restricted(sig, HZ, Hnu_Z, LZ, bw=bw,ranges=ranges)

    # Define initial values
    x_init = [Lnu_Z.sqrt()]
    print("x_init ", x_init)
    print(x_init[0].type())

    dxmax = [x_init[0].mean()]
    print(dxmax[0].type())

    def callback_restricted(xu):
        nZ = LZ.detach().cpu().numpy()
        nnu_Z = xu[0].detach().cpu().numpy() ** 2
        np.savez_compressed(opath + outfile + '_restricted', Z=nZ, nu_Z=nnu_Z)
        return nZ, nnu_Z

    Z, nu_Z = optimize(L_restricted.loss, x_init, dxmax, nb_iter=10, callback=callback_restricted)
    writeParticleVTK(opath+outfile+'_restricted.npz',featNames=genes)

    ## Second step ##

    # Define loss functions
    L_all = ParticleLoss_full(sig, HZ, Hnu_Z, bw=bw,ranges=ranges)

    # Define initial values
    x_init = [torch.tensor(Z).type(dtype), torch.tensor(nu_Z).type(dtype).sqrt()]
    print("x_init ", x_init)

    dxmax = [sig, x_init[1].mean()]
    print(dxmax)

    def callback_all(xu):
        nZ = xu[0].detach().cpu().numpy()
        nnu_Z = xu[1].detach().cpu().numpy() ** 2
        np.savez_compressed(opath + outfile + '_all', Z=nZ, nu_Z=nnu_Z)
        return nZ, nnu_Z

    optimize(L_all.loss, x_init, dxmax, nb_iter=20, callback=callback_all)
    writeParticleVTK(opath+outfile+'_all.npz',condense=False,featNames=genes)
