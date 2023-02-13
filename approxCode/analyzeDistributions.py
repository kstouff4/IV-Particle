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

def selectFeatures(particleNPZ,listOflist,listOfNames,inverse=False):
    '''
    Write out separate vtk file using name in list
    
    e.g. ERC = [10, 14, 23, 70, 250, 258, 329, 361, 368, 585]
         Sub = [241,421, 531, 559, 641]
         CA1 = [197]
         CA2 = [208]
         CA3 = [230]
         DG = [310, 423, 593, 594]
    '''
    npz = np.load(particleNPZ,allow_pickle=True)
    nuZ = npz[npz.files[1]]
    Z = npz[npz.files[0]]
    indsTot = nuZ[:,0] < 0
    indsTotList = 'not_'
    for n in range(len(listOfNames)):
        indsTotList += listOfNames[n]
        l = listOflist[n]
        l[0]
        inds = nuZ[:,int(l[0])] > 0.0001
        for i in range(1,len(l)):
            inds += nuZ[:,int(l[i])] > 0.0001
        Znew = Z[inds,:]
        nuZnew = nuZ[inds,:]
        if np.sum(inds) < 1:
            continue
        np.savez(particleNPZ.replace('.npz','_' + listOfNames[n] + '.npz'),Z=Znew,nu_Z=nuZnew)
        maxInd = np.argmax(nuZnew,axis=-1)+1
        vtf.writeVTK(Znew,[maxInd,np.sum(nuZnew,axis=-1)],['MAX_VAL_NU','TOTAL_MASS'],particleNPZ.replace('.npz','_' + listOfNames[n] + '.vtk'),polyData=None)
        indsTot += inds # keeps track of indices in one of any compartments in list
    if (inverse):
        Zinv = Z[indsTot == 0,:]
        nuZinv = nuZ[indsTot == 0,:]
        np.savez(particleNPZ.replace('.npz','_' + indsTotList + '.npz'),Z=Zinv,nu_Z=nuZinv)
        maxInd = np.argmax(nuZinv,axis=-1)+1
        vtf.writeVTK(Zinv,[maxInd,np.sum(nuZinv,axis=-1)],['MAX_VAL_NU','TOTAL_MASS'],particleNPZ.replace('.npz','_' + indsTotList + '.vtk'),polyData=None)
    
    return
    
def selectPlanes(particleNPZ,thick=0.05,ax=2,s=None,e=None):
    '''
    Select Planes of full particle object based on coordinates
    '''
    npz = np.load(particleNPZ)
    nuZ = npz[npz.files[1]]
    Z = npz[npz.files[0]]
    
    if (s is None):
        ma = np.max(Z,axis=(0))
        mi = np.min(Z,axis=(0))
        totSections = np.ceil((ma - mi)/thick)
        totSections = int(totSections[ax])
        for t in range(totSections):
            inds = (Z[:,ax] > (mi[ax] + t*thick))*(Z[:,ax] <= (mi[ax] + (t+1)*thick))
            if np.sum(inds) < 1:
                continue
            Znew = Z[inds,:]
            nuZnew = nuZ[inds,:]
            np.savez(particleNPZ.replace('.npz','_' + str(t) + 'outof' + str(totSections) + 'along' + str(ax) + '.npz'),Z=Znew,nu_Z=nuZnew)
            maxInd = np.argmax(nuZnew,axis=-1)+1
            vtf.writeVTK(Znew,[maxInd,np.sum(nuZnew,axis=-1)],['MAX_VAL_NU','TOTAL_MASS'],particleNPZ.replace('.npz','_' + str(t) + 'outof' + str(totSections) + 'along' + str(ax) + '.vtk'),polyData=None)
    else:
        sInd = s
        while (sInd + thick <= e):
            inds = (Z[:,ax] > sInd) * (Z[:,ax] <= sInd + thick)
            if np.sum(inds) < 1:
                sInd += thick
                continue
            Znew = Z[inds,:]
            nuZnew = nuZ[inds,:]
            np.savez(particleNPZ.replace('.npz','_' + str(sInd) + 'to' + str(sInd+thick) + 'along' + str(ax) + '.npz'),Z=Znew,nu_Z=nuZnew)
            maxInd = np.argmax(nuZnew,axis=-1)+1
            vtf.writeVTK(Znew,[maxInd,np.sum(nuZnew,axis=-1)],['MAX_VAL_NU','TOTAL_MASS'],particleNPZ.replace('.npz','_' + str(sInd) + 'to' + str(sInd+thick) + 'along' + str(ax) + '.vtk'),polyData=None)
            sInd += thick
            
    return
