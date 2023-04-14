import numpy as np
import os
import sys
from sys import path as sys_path
import glob

sys_path.append('/cis/home/kstouff4/Documents/SurfaceTools/')
import vtkFunctions as vtf

import subsampleFunctions as ssf
from approximateIntermediate import *

def main():
    sig = 0.1
    nb_iter0=7
    nb_iter1=20
    Nmax=1500.0
    Npart=2000.0

    xInter = '/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenMerfish/ZApprox_sig0.05Uniform_Aligned/'
    outpath = '/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenMerfish/ZApprox_sig0.1All/'
    if (not os.path.exists(outpath)):
        os.mkdir(outpath) 

    original = sys.stdout
    sys.stdout = open(outpath+'output.txt','w')
    
    filsX = glob.glob(xInter + '*.npz')
    
    # combine files
    info = np.load(filsX[0])
    Xt = info['Z']
    X = Xt - np.mean(Xt,axis=0) # center all data 
    nuX = info['nu_Z']
    
    
    for f in range(1,len(filsX)):
        info = np.load(filsX[f])
        Xt = info['Z']
        Xt = Xt - np.mean(Xt,axis=0)
        X = np.vstack((X,Xt))
        nuX = np.vstack((nuX,info['nu_Z']))
    np.savez(xInter + 'allSlicesCentered.npz',X=X,nu_X=nuX)
    
    fZ = ssf.makeSubsample(xInter + 'allSlicesCentered.npz',sig,xInter + 'allSlicesCentered',xtype='semi-discrete',ztype='semi-discrete',overhead=0.1,maxV=nuX.shape[-1],C=1.2)
    
    project3D(xInter + 'allSlicesCentered.npz', sig, nb_iter0, nb_iter1,outpath,Nmax=Nmax,Npart=Npart,Zfile=fZ,maxV=nuX.shape[-1],optMethod='LBFGS',C=1.2)
    
    sys.stdout = original
    
    return

if __name__ == "__main__":
    main()