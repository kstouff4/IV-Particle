import os
import sys
from sys import path as sys_path
sys_path.append('../')

from varap.io.load_BarSeqCells import BarSeqCellLoader
from varap.loss.particles import ParticleLoss_full, ParticleLoss_restricted, ParticleLoss_ranges
from varap.optim.band import optimize
from varap.io.writeOut import writeParticleVTK

import numpy as np
import torch 
import glob

dtype = torch.cuda.FloatTensor
torch.set_default_tensor_type(dtype)

dataDir = sys.argv[1]

# fpath = '/content/drive/My Drive/Kaitlin/'
fpathO = '/cis/home/kstouff4/Documents/MeshRegistration/Particles/BarSeq/top28MI/sig0.025/subsampledObject.pkl'
fpath = '/cis/home/kstouff4/Documents/MeshRegistration/Particles/BarSeqAligned/top28MI/sig0.1_dimEff2/initialHighLowAll.pkl'
opath = '/cis/home/kstouff4/Documents/MeshRegistration/Particles/BarSeqAligned/top28MI/sig0.2/'

fpath = '/cis/home/kstouff4/Documents/MeshRegistration/ParticleLDDMMQP/sandbox/SliceToSlice/BarSeqAligned/Half_Brain_D079/sig0.25Align_HighRes/Whole/initialHighLowAllCells_nu_R.pkl'
opath = '/cis/home/kstouff4/Documents/MeshRegistration/ParticleLDDMMQP/sandbox/SliceToSlice/BarSeqAligned/Half_Brain_D079/sig0.25Align_200um/Whole/'

fpath = '/cis/home/kstouff4/Documents/MeshRegistration/ParticleLDDMMQP/sandbox/SliceToSlice/BarSeq/HalfBrains/' + dataDir + '/0.25/initialHighLowAllCells_nu_R.pkl'
opath = '/cis/home/kstouff4/Documents/MeshRegistration/ParticleLDDMMQP/sandbox/SliceToSlice/BarSeq/HalfBrains/' + dataDir + '/approx/0.1/'

os.makedirs(opath,exist_ok=True)

bw = 10
sig = 0.1
optimizeAll = False

a = BarSeqCellLoader(fpath,[0.0,0.0,0.200])

## First step ##

# Define loss functions
numFiles = a.getNumberOfFiles()
print("number of files optimizing: ", numFiles)

def combineFiles(filepath):
    fils = glob.glob(filepath + '*optimal_all.npz')
    info = np.load(fils[0])
    X = info[info.files[0]]
    nu_X = info[info.files[1]]
    
    for i in range(1,len(fils)):
        info = np.load(fils[i])
        X = np.vstack((X,info[info.files[0]]))
        nu_X = np.vstack((nu_X,info[info.files[1]]))
    
    np.savez(filepath + 'allCombined_optimal_all.npz',X=X,nu_X=nu_X)
    writeParticleVTK(filepath + 'allCombined_optimal_all.npz')
    return

def optimizeFunc(HZi,Hnu_Zi,LZi,Lnu_Zi,outfile):
    ranges = ParticleLoss_ranges(sig, HZi, Hnu_Zi, LZi, Lnu_Zi)
    print(torch.allclose(LZi,ranges.Z))

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
    writeParticleVTK(opath+outfile+'_restricted.npz',featNames=a.featNames)

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
    writeParticleVTK(opath+outfile+'_all.npz',condense=False,featNames=a.featNames)
    return

if optimizeAll:
    nHZ,nHnu_Z,nLZ,nLnu_Z = a.getHighLowPair(0)
    HZ = torch.tensor(nHZ).type(dtype)
    Hnu_Z = torch.tensor(nHnu_Z).type(dtype)
    LZ = torch.tensor(nLZ).type(dtype)
    Lnu_Z = torch.tensor(nLnu_Z).type(dtype) 
    print(HZ.shape)
    print(Hnu_Z.shape)
    print(LZ.shape)
    print(Lnu_Z.shape)
    for f in range(1,numFiles):
        nHZ,nHnu_Z,nLZ,nLnu_Z = a.getHighLowPair(f)
        HZ = torch.cat((HZ,torch.tensor(nHZ).type(dtype)))
        Hnu_Z = torch.cat((Hnu_Z,torch.tensor(nHnu_Z).type(dtype)))
        LZ = torch.cat((LZ,torch.tensor(nLZ).type(dtype)))
        Lnu_Z = torch.cat((Lnu_Z,torch.tensor(nLnu_Z).type(dtype)))
    
    print(HZ.shape)
    print(Hnu_Z.shape)
    print(LZ.shape)
    print(Lnu_Z.shape)
    outfile = 'all_optimal'
    optimizeFunc(HZ,Hnu_Z,LZ,Lnu_Z,outfile)
else:
    '''
    for f in range(numFiles):
        nHZ,nHnu_Z,nLZ,nLnu_Z = a.getHighLowPair(f)
        print(np.unique(nHZ[:,-1]))
        print(np.unique(nLZ[:,-1]))
        HZ = torch.tensor(nHZ).type(dtype)
        Hnu_Z = torch.tensor(nHnu_Z).type(dtype)
        LZ = torch.tensor(nLZ).type(dtype)
        Lnu_Z = torch.tensor(nLnu_Z).type(dtype)

        outfile = a.getFilename_subsample(f).split('/')[-1].replace('.npz','_optimal')
        optimizeFunc(HZ,Hnu_Z,LZ,Lnu_Z,outfile)
     '''
    combineFiles(opath)
    