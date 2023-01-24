import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sys import path as sys_path
sys_path.append('/cis/home/kstouff4/Documents/SurfaceTools/')
import vtkFunctions as vtf

#################################################

def getEntropyParticlesNuX(particleNPZ):
    '''
    Compute the entropy of distribution for each of the particles.
    Plot histogram of entropy values for particles.
    Save entropy as part of npz file 
    '''
    
    npz = np.load(particleNPZ)
    nuZ = npz['nu_Z']
    nuZN = nuZ / (np.sum(nuZ,axis=-1)[...,None])
    h = nuZN*np.log(nuZN,where=(nuZN > 0))
    h = np.sum(h,axis=-1)
    h = -h
    
    f,ax = plt.subplots()
    ax.hist(h)
    f.savefig(particleNPZ.replace('.npz','_entropyHist.png'),dpi=300)
    
    np.savez(particleNPZ.replace('.npz','_entropy.npz'),h=h)
    maxInd = np.argmax(nuZ,axis=-1)+1
    vtf.writeVTK(npz['Z'],[maxInd,np.sum(nuZ,axis=-1),h],['MAX_VAL_NU','TOTAL_MASS','ENTROPY'],particleNPZ.replace('.npz','_entropy.vtk'),polyData=None)
    return

def getEntropyEstimated(thetaNPZ):
    '''
    Assume theta is pi_theta = estimation of alpha_c*zeta_c in target
    Find entropy by normalizing 
    '''
    
    npz = np.load(thetaNPZ)
    theta = npz['theta']
    ntheta = theta - np.min(theta,axis=-1)[...,None] # correct for negative values that shouldn't exist 
    nntheta = ntheta / (np.sum(ntheta,axis=-1)[...,None])
    h = nntheta*np.log(nntheta,where=(nntheta > 0))
    h = np.sum(h,axis=-1)
    h = -h
    
    f,ax = plt.subplots()
    ax.hist(h)
    f.savefig(thetaNPZ.replace('.npz','_entropyHist.png'),dpi=300)
    np.savez(thetaNPZ.replace('.npz','_entropy.npz'),h=h)
    
    return

