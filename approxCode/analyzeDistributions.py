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

def selectGenes(npzFile,geneInds,geneNames,savename,merge=False,maxVal=113,d=3):
    '''
    npzFile = with X and nu_X or Z and nu_Z where nu_X/nu_Z is either single number indicating gene ind or is N x L array
    merge = True if npzFile is list and you want to read in all of them from list and merge genes together
    
    example (Barseq): 
    npzFile = glob.glob('/cis/home/kstouff4/Documents/SpatialTranscriptomics/BarSeq/Genes/slice[0-9][0-9].npz')
    geneInds = [28,56,111,101,47]
    geneNames = ['Rab3c','Gria1','Slc17a7','Dgkb','Nrsn1']
    savename = '/cis/home/kstouff4/Documents/SpatialTranscriptomics/BarSeq/MI_ResultsGenes/5highMI.npz'
    merge = True
    maxVal=113
    '''
    
    if (merge):
        npzF0 = npzFile[0]
    else:
        npzF0 = npzFile
    
    info = np.load(npzF0)
    Z = info[info.files[0]]
    nuZ = info[info.files[1]]
    if (len(info.files) > 2):
        nuZc = info[info.files[2]]
    
    Zs = []
    nuZs = []
    nuZcs = []
    tot = 0
    if (len(nuZ.shape) < 2 or nuZ.shape[-1] == 1):
        if (np.min(nuZ) > 0):
            nuZ = nuZ - 1
        for ge in geneInds:
            inds = nuZ == ge
            print(inds.shape)
            Zs.append(Z[np.squeeze(inds),...])
            nuZs.append(nuZ[inds])
            nuZcs.append(nuZc[inds])
            tot += np.sum(inds)
        ZsTot = np.zeros((tot,d))
        nuZsTot = np.zeros((tot,1))
        nuZcsTot = np.zeros((tot,1))
        cnt = 0
        for i in range(len(Zs)):
            ZsTot[cnt:Zs[i].shape[0]+cnt,:] = Zs[i]
            nuZsTot[cnt:nuZs[i].shape[0]+cnt] = nuZs[i][...,None]
            nuZcsTot[cnt:nuZcs[i].shape[0]+cnt] = nuZcs[i][...,None]
            cnt += Zs[i].shape[0]
    np.savez(npzFile.replace('.npz','_' + str(geneInds) + '.npz'),X=ZsTot,nu_X=nuZsTot)
    vtf.writeVTK(ZsTot,[nuZsTot,nuZcsTot],['GeneID','CellID'],npzFile.replace('.npz','_' + str(geneInds) + '.vtk'),polyData=None)
    
    return

def mergeSlices(files,savename,d=3,center=False):
    Z = []
    nuZ = []
    totNum = 0
    for f in files:
        info = np.load(f)
        x = info['X']
        nux = info['nu_X']
        if (center):
            minx = np.min(x,axis=0)
            maxx = np.max(x,axis=0)
            meanx = 0.5*(maxx + minx)
            x = x - meanx
        Z.append(x)
        nuZ.append(nux)
        totNum += x.shape[0]
    print("total number is, ", totNum)
    Zar = np.zeros((totNum,d))
    nuZar = np.zeros((totNum,1))
    cnt = 0
    for z in range(len(Z)):
        Zar[cnt:Z[z].shape[0]+cnt,...] = Z[z]
        nuZar[cnt:nuZ[z].shape[0]+cnt] = nuZ[z]
        cnt+=Z[z].shape[0]
    np.savez(savename,X=Zar,nu_X=nuZar)
    np.savez(savename.replace('.npz','_mm.npz'),X=Zar/1000.0,nu_X=nuZar)
    return
                                           
                           