import time
import sys
from sys import path as sys_path
sys_path.append('/cis/home/kstouff4/Documents/SurfaceTools/')
import vtkFunctions as vtf

import torch
import numpy as np
import numpy.matlib

from matplotlib import pyplot as plt

from pykeops.torch import LazyTensor
import pykeops.config


from pykeops.torch import Vi, Vj
#from pykeops.torch import LazyTensor

from pykeops.torch.cluster import sort_clusters
from pykeops.torch.cluster import cluster_ranges_centroids
from pykeops.torch.cluster import grid_cluster
from pykeops.torch.cluster import from_matrix

np_dtype = "float32"
dtype = torch.cuda.FloatTensor 

import os, psutil
import pykeops
pykeops.clean_pykeops()
torch.cuda.empty_cache()
import GPUtil
GPUtil.showUtilization()

############################################################################
# Helper Functions for dealing with Large Feature Spaces

def oneHot(nu_X,nu_Z):
    '''
    Make nu_X into full nu_X by expanding single dimension to maximum number
    Make nu_Z into subsampled nu_Z 
    '''
    nnu_X = np.zeros((nu_X.shape[0],nu_Z.shape[-1])).astype('bool_') # assume nu_Z has full spectrum
    nnu_X[np.arange(nu_X.shape[0]),np.squeeze(nu_X-1).astype(int)] = 1
    print(np.unique(nu_X))
    
    nonZeros = np.sum(nnu_X,axis=0)+np.sum(nu_Z,axis=0)
    indsToKeep = np.where(nonZeros > 0)
    print("total is " + str(len(indsToKeep[0]))) # 0 based with maximum = 1 less than dimension
    print(indsToKeep[0])
    
    nnu_X = nnu_X[:,indsToKeep[0]]
    nnu_Z = nu_Z[:,indsToKeep[0]]

    return nnu_X,nnu_Z,indsToKeep[0]

def makeOneHot(nu,maxVal=673):
    nu1 = np.zeros((nu.shape[0],maxVal)).astype('float32')
    nu1[np.arange(nu.shape[0]),np.squeeze(nu-1).astype(int)] = 1
    return nu1

def getXZ(npzX,npzZ):
    '''
    Assume initial Z is already in one hot encoding for features
    '''
    xInfo = np.load(npzX)
    X = xInfo['X']
    print("numbers of X: " + str(X.shape[0]))
    zInfo = np.load(npzZ)
    Z = zInfo['Z']
    print("numbers of Z: " + str(Z.shape[0]))
    nu_X, nu_Z,indsToKeep = oneHot(xInfo['nu_X'],zInfo['nu_Z'])
    Xlist = groupXbyLabel(nu_X,X)

    return Xlist,Z,nu_Z,indsToKeep,X,nu_X

def returnToInt(oneHotEncode):
    d = oneHotEncode.shape[-1]
    x = np.argmax(oneHotEncode,axis=-1)
    return x[...,None],d

def groupXbyLabel(oneHotEncode,X):
    d = oneHotEncode.shape[-1]
    listOfX = []
    for i in range(d):
        listOfX.append(X[oneHotEncode[:,i]])
    return listOfX
                           

def getXZInt(npzX,npzZ):
    '''
    Assume initial Z is already in one hot encoding for features
    Encode X with number - 1 to denote integer of 1 
    '''
    xInfo = np.load(npzX)
    X = xInfo['X']
    zInfo = np.load(npzZ)
    Z = zInfo['Z']
    nu_X, nu_Z,indsToKeep = oneHot(xInfo['nu_X'],zInfo['nu_Z'])
    
    # nu_X is now in one HOT for reduced set 
    nu_X, d = returnToInt(nu_X)
    return X,nu_X,Z,nu_Z,indsToKeep,d
                        

def project3D(Xfile, sigma, nb_iter0, nb_iter1,outpath,Nmax=2000.0,Npart=50000.0,Zfile=None,maxV=673,optMethod='LBFGS'):
    '''
    Find and optimize subsample of points for X,nu_X defined by volume of X and sigma
    
    eps = denotes size of blocks for block sparse reduction
    sigma = sigma of space kernel 
    nb_iter0 = (2)
    nb_iter1 = (30)
    outpath = where to save results 
    Z, nu_Z = if already computed Z's
    
    Yongsoo max = 351
    Allen max = 673
    
    OptMethods = LBFGS or Adam (less memory intensive)
    '''
    C=1.2
    sig = sigma
    lossTrack = []
    
    process = psutil.Process(os.getpid())
    #print("memory (bytes)")
    #print(process.memory_info().rss)  # in bytes

    '''
    NOT USEFUL FOR BY LABEL SAMPLING
    '''
    def make_subsample(X, nu_X):
        # Input:
        # X is the initial data
        # N is the number of random subsample

        # Output 
        # Y is the output subsample
        volBB = np.prod(np.max(X,axis=0)-np.min(X,axis=0))
        N = np.round(C*volBB/(sig**3)).astype(int)
        print("N " + str(N) + " vs " + str(X.shape[0]))
        N = min(X.shape[0],N)
        sub_ind = np.random.choice(X.shape[0],replace = False, size = N)
        print(sub_ind.shape)
        Z = X[sub_ind,:]
        nu_Z = nu_X[sub_ind,:] # start weights off as fraction of what you started with
        return Z, makeOneHot(nu_Z,maxVal=maxV)*X.shape[0]/N
    
    def make_ranges(Xlist, Z, epsX, epsZ,sig):
      # Here X and Z are torch tensors
        a = np.sqrt(3)
        rangesXXList = []
        rangesZXList = []
        X_labelsList = []
        Z_labels = grid_cluster(Z, epsZ) 
        Z_ranges, Z_centroids, _ = cluster_ranges_centroids(Z, Z_labels)
        D = ((Z_centroids[:, None, :] - Z_centroids[None, :, :]) ** 2).sum(dim=2)
        keep = D <(a*epsZ+4* sig) ** 2
        rangesZZ_ij = from_matrix(Z_ranges, Z_ranges, keep)
        areas = (Z_ranges[:, 1] - Z_ranges[:, 0])[:, None] * (Z_ranges[:, 1] 
                        - Z_ranges[:, 0])[None, :]
        total_area = areas.sum()  # should be equal to N*M
        sparse_area = areas[keep].sum()
        print(
        "We keep {:.2e}/{:.2e} = {:2d}% of the original kernel matrix.".format(
          sparse_area, total_area, int(100 * sparse_area / total_area)))
        print("")

        for cc in range(len(Xlist)):
            X = Xlist[cc]
            eps = epsX[cc]
            X_labels = grid_cluster(X, eps)
            X_labelsList.append(X_labels)
            X_ranges, X_centroids, _ = cluster_ranges_centroids(X, X_labels)
            D = ((X_centroids[:, None, :] - X_centroids[None, :, :]) ** 2).sum(dim=2)
            a = np.sqrt(3)
            keep = D <(a*eps+4* sig) ** 2
            rangesXX_ij = from_matrix(X_ranges, X_ranges, keep)
            areas = (X_ranges[:, 1] - X_ranges[:, 0])[:, None] * (X_ranges[:, 1] 
                            - X_ranges[:, 0])[None, :]
            total_area = areas.sum()  # should be equal to N*M
            sparse_area = areas[keep].sum()
            print(
            "We keep {:.2e}/{:.2e} = {:2d}% of the original kernel matrix.".format(
              sparse_area, total_area, int(100 * sparse_area / total_area)))
            print("")

            D = ((Z_centroids[:, None, :] - X_centroids[None, :, :]) ** 2).sum(dim=2)
            keep = D <(a*eps+4* sig) ** 2
            rangesZX_ij = from_matrix(Z_ranges, X_ranges, keep)
            areas = (Z_ranges[:, 1] - Z_ranges[:, 0])[:, None] * (X_ranges[:, 1] 
                            - X_ranges[:, 0])[None, :]
            total_area = areas.sum()  # should be equal to N*M
            sparse_area = areas[keep].sum()
            print(
            "We keep {:.2e}/{:.2e} = {:2d}% of the original kernel matrix.".format(
              sparse_area, total_area, int(100 * sparse_area / total_area)))
            print("")
            #print("memory (bytes)")
            #print(process.memory_info().rss)  # in bytes
            rangesXXList.append(rangesXX_ij)
            rangesZXList.append(rangesZX_ij)
            
        return rangesXXList, rangesZZ_ij, rangesZXList, X_labelsList, Z_labels

    def make_loss(tXlist, len_Z, dim_nu_Z, rangesXX, rangesZZ_ij, rangesZX):
        c = 0
        for lab in range(len(tXlist)):
            rangesXX_ij = rangesXX[lab]
            rangesZX_ij = rangesZX[lab] # Getting the ranges
            tX = tXlist[lab]
            LX_i = LazyTensor(tX[:,None,:])
            LX_j = LazyTensor(tX[None,:,:])

            D_ij = ((LX_i - LX_j)**2/sig**2).sum(dim=2)  
            K_ij = (- D_ij).exp()
            K_ij.ranges = rangesXX_ij
            c +=  K_ij.sum(dim=1).sum()
            print('c=',c.detach())

        def loss(tZal_Z):
            # z operation is still assumed to be the same (mixed distribution of labels)
            LZ_i, LZ_j = Vi(tZal_Z[0:3*len_Z].view(-1,3)), Vj(tZal_Z[0:3*len_Z].view(-1,3))
    
            Lnu_Z_i= Vi(tZal_Z[3*len_Z::].view(-1,dim_nu_Z)**2)
            Lnu_Z_j= Vj(tZal_Z[3*len_Z::].view(-1,dim_nu_Z)**2)

            DZZ_ij = ((LZ_i - LZ_j)**2/sig**2).sum(dim=2)  
            KZZ_ij = (- DZZ_ij).exp()  
            PZZ_ij = (Lnu_Z_i*Lnu_Z_j).sum(dim=2)
            KPZZ_ij = KZZ_ij*PZZ_ij
            KPZZ_ij.ranges = rangesZZ_ij
            
            # only compare Z's that share the same support (intersection > 0)
            #for lab in range(len(tXlist)):
                
            
            L = KPZZ_ij.sum(dim=1).sum()
            print("shape of L is ")
            print(L.shape)
            torch.cuda.empty_cache()
            print("memory (bytes)")
            print(process.memory_info().rss)  # in bytes
            GPUtil.showUtilization()
            L.backward()
            print("memory (bytes)")
            print(process.memory_info().rss)  # in bytes
            GPUtil.showUtilization()
            for lab in range(len(tXlist)):
                rangesXX_ij = rangesXX[lab]
                rangesZX_ij = rangesZX[lab] # Getting the ranges
                tX = tXlist[lab]
                #LX_i = LazyTensor(tX[:,None,:])
                LX_j = LazyTensor(tX[None,:,:])
                DZX_ij = ((LZ_i - LX_j)**2/sig**2).sum(dim=2) 
                KZX_ij = (- DZX_ij).exp() 
            #PZX_ij = (Lnu_Z_i*Lnu_X_j).sum(dim=2) # 
                KPZX_ij = Vi(tZal_Z[3*len_Z::].view(-1,dim_nu_Z)[...,lab][...,None]**2)*KZX_ij
                # KPZX_ij = Lnu_Z_i.elem(lab)*KZX_ij # Alternative 
                KPZX_ij.ranges = rangesZX_ij
            
                #selection = (tZal_Z[3*len_Z::].view(-1,dim_nu_Z)**2).astype('float32')
                #Ef = selection[:,lab]*(KPZZ_ij.sum(dim=1))
                Es = -2*KPZX_ij.sum(dim=1).sum()
                Es.backward()
                #print("passed Es loss all for " + str(lab) + " out of " + str(len(tXlist)-1))
                L += Es
                #del LX_i, LX_j, DZX_ij, KZX_ij, KPZX_ij
            L += c
            return L.detach(),c.detach()
        return loss 
    
    def make_loss2(tXlist, len_Z, dim_nu_Z, rangesXX, rangesZZ_ij, rangesZX):
        c = 0
        for lab in range(len(tXlist)):
            rangesXX_ij = rangesXX[lab]
            rangesZX_ij = rangesZX[lab] # Getting the ranges
            tX = tXlist[lab]
            LX_i = LazyTensor(tX[:,None,:])
            LX_j = LazyTensor(tX[None,:,:])

            D_ij = ((LX_i - LX_j)**2/sig**2).sum(dim=2)  
            K_ij = (- D_ij).exp()
            K_ij.ranges = rangesXX_ij
            c +=  K_ij.sum(dim=1).sum()
            print('c=',c.detach())

        LZ_i, LZ_j= Vi(tZ), Vj(tZ)
        DZZ_ij = ((LZ_i - LZ_j)**2/sig**2).sum(dim=2)  
        KZZ_ij = (- DZZ_ij).exp()    

        def loss(tal_Z): 
            Lnu_Z_i, Lnu_Z_j = Vi(tal_Z.view(-1,dim_nu_Z)**2), Vj(tal_Z.view(-1,dim_nu_Z)**2)

            PZZ_ij = (Lnu_Z_i*Lnu_Z_j).sum(dim=2)
            KPZZ_ij = KZZ_ij*PZZ_ij
            KPZZ_ij.ranges = rangesZZ_ij
            
            L = KPZZ_ij.sum(dim=1).sum()
            print("shape of L is ")
            print(L.shape)
            #L.backward() # just keep Lnu_Z_i's
            #KPZZ_ij = None
            #PZZ_ij = None
            #Lnu_Z_j = None
            #print("passed L loss")
            torch.cuda.empty_cache()
            print("memory (bytes)")
            print(process.memory_info().rss)  # in bytes
            GPUtil.showUtilization()
            L.backward()
            print("memory (bytes)")
            print(process.memory_info().rss)  # in bytes
            GPUtil.showUtilization()
            for lab in range(len(tXlist)):
                rangesXX_ij = rangesXX[lab]
                rangesZX_ij = rangesZX[lab] # Getting the ranges
                tX = tXlist[lab]
                #LX_i = LazyTensor(tX[:,None,:])
                LX_j = LazyTensor(tX[None,:,:])
                DZX_ij = ((LZ_i - LX_j)**2/sig**2).sum(dim=2) 
                KZX_ij = (- DZX_ij).exp() 
            #PZX_ij = (Lnu_Z_i*Lnu_X_j).sum(dim=2) # 
                KPZX_ij = Vi(tal_Z.view(-1,dim_nu_Z)[...,lab][...,None]**2)*KZX_ij
                #KPZX_ij = Lnu_Z_i.elem(lab)*KZX_ij # alternative 
                KPZX_ij.ranges = rangesZX_ij
            
                #selection = (tal_Z.view(-1,dim_nu_Z)**2).astype('float32')
                #print("selection size is " + str(selection.shape))
                #print("memory (bytes)")
                #print(process.memory_info().rss)  # in bytes
                #ss = selection[:,lab]
                #print("ss size is " + str(ss.shape))
                #Ef = selection[:,lab]*(KPZZ_ij.sum(dim=1))
                Es = -2*KPZX_ij.sum(dim=1).sum()
                L += Es
                Es.backward() # deleting retain graph                 
                #print("passed Es loss for " + str(lab) + " out of " + str(len(tXlist)-1))
                #Lback.append(E.sum().backward())
            L += c
            return L.detach(),c.detach()
        print("memory (bytes)")
        print(process.memory_info().rss)  # in bytes
        return loss
    
    # get X and Z
    if (Zfile is not None):
        Xlist,Z,nu_Z,indsToKeep,X,nu_X = getXZ(Xfile,Zfile)
    else:
        xInfo = np.load(Xfile)
        X = xInfo['X']
        nu_X = xInfo['nu_X']
        Z, nu_Z = make_subsample(X,nu_X) # returns Z as one hot encoding
        nu_X, nu_Z,indsToKeep = oneHot(nu_X,nu_Z)
        Xlist = groupXbyLabel(nu_X,X)
        print("number in new Z")
        print(Z.shape[0])
        
        # write original Z 
        maxInd = np.argmax(nu_Z,axis=-1)+1
        vtf.writeVTK(Z,[maxInd,np.sum(nu_Z,axis=-1)],['MAX_VAL_NU','TOTAL_MASS'],outpath+'_originalZnu_ZwC' + str(C) + '_sig' + str(sig) + '.vtk',polyData=None)
        np.savez(outpath+'_originalZnu_ZwC' + str(C) + '_sig' + str(sig) + '.npz',Z=Z, nu_Z=nu_Z)

    # make Epislon List
    def makeEps(Xlist,Z,Nmax,Npart):
        '''
        Returns list of epsilons to use for making ranges
        '''
        epsList = []
        volList = []
        partList = []
        ncubeList = []
        denZ = min(Z.shape[0]/Npart,Nmax)
        volZ = np.prod(np.max(Z,axis=0) - np.min(Z,axis=0) + 0.001) # volume of bounding box (avoid 0); 1 micron
        epsZ = np.cbrt(volZ/denZ)
        print("ZX\tVol\tParts\tCubes\tEps")
        print("Z\t" + str(volZ) + "\t" + str(Z.shape[0]) + "\t" + str(denZ) + "\t" + str(epsZ))
        for X in Xlist:
            den = min(X.shape[0]/Npart,Nmax)
            volX = np.prod(np.max(X,axis=0)-np.min(X,axis=0) + 0.001)
            epsX = np.cbrt(volX/den)
            epsList.append(epsX)
            print("X\t" + str(volX) + "\t" + str(X.shape[0]) + "\t" + str(den) + "\t" + str(epsX))
        return epsList,epsZ
        
    epsX,epsZ = makeEps(Xlist,Z,Nmax,Npart)
    print("epsilons are")
    print(epsX)
    print(epsZ)
    
    # make tensors for each
    tXlist = []
    for Xl in Xlist:
        tXlist.append(torch.tensor(Xl).type(dtype))
        
    tZ = torch.tensor(Z).type(dtype)
    tnu_Z = torch.tensor(nu_Z).type(dtype)

    # Computes ranges and labels for the grid
    print("Making ranges")
    rangesXXList, rangesZZ_ij, rangesZXList, X_labelsList, Z_labels = make_ranges(tXlist, tZ, epsX, epsZ,sig)

    # Sorts X and nu_X
    print("Sorting X and nu_X")
    for k in range(len(tXlist)):
        tXlist[k], _ = sort_clusters(tXlist[k], X_labelsList[k]) # sorting the labels

    #  Sorts Z and nu_Z
    print("Sorting Z and nu_Z")
    tZ, _ = sort_clusters(tZ, Z_labels) # sorting the labels
    tnu_Z, _ = sort_clusters(tnu_Z, Z_labels)
    len_Z, dim_nu_Z = nu_Z.shape

    # Optimization

    def optimize(tZ, tnu_Z, nb_iter = 20, flag = 'all'):
        if flag == 'all':
            loss = make_loss(tXlist, len_Z, dim_nu_Z, rangesXXList, rangesZZ_ij, rangesZXList)
            p0 = torch.cat((tZ.flatten(),tnu_Z.pow(0.5).flatten()),0).requires_grad_(True)
        else:
            loss = make_loss2(tXlist, tZ, dim_nu_Z, rangesXXList, rangesZZ_ij, rangesZXList)
            p0 = tnu_Z.pow(0.5).flatten().clone().requires_grad_(True)

        print('p0', p0.is_contiguous())
        if (optMethod == 'LBFGS'):
            optimizer = torch.optim.LBFGS([p0], max_eval=10, max_iter=10, line_search_fn = 'strong_wolfe',history_size=3)
        elif (optMethod == 'Adam'):
            optimizer = torch.optim.Adam([p0])
        else:
            print("optimizing method is not supported. defaulting to LBFGS")
            optimizer = torch.optim.LBFGS([p0], max_eval=10, max_iter=10, line_search_fn = 'strong_wolfe',history_size=3)
        
        def closure():
            optimizer.zero_grad(set_to_none=True)
            L,c = loss(p0)
            print("error is ", L.detach().cpu().numpy())
            print("relative error loss", L.detach().cpu().numpy()/c.detach().cpu().numpy())
            #lossTrack.append(L.detach().cpu().numpy()/c.detach().cpu().numpy())
            #tot.backward()
            return L

        for i in range(nb_iter):
            print("it ", i, ": ", end="")
            print(torch.cuda.memory_allocated(0))
            GPUtil.showUtilization()
            optimizer.step(closure)
            #torch.cuda.empty_cache()

        if flag == 'all':
            tnZ = p0[0:3*len_Z].detach().view(-1,3)
            tnnu_Z = p0[3*len_Z::].detach().view(-1,dim_nu_Z)**2
        else:
            tnZ = tZ
            tnnu_Z = p0.detach().view(-1,dim_nu_Z)**2
        return tnZ, tnnu_Z

    print("Starting Optim")
    print("sum tnu_Z before", tnu_Z.sum())
    tZ, tnu_Z = optimize(tZ, tnu_Z, nb_iter = nb_iter0, flag = '')
    tnZ, tnnu_Z = tZ, tnu_Z
    torch.cuda.empty_cache()
    tnZ, tnnu_Z = optimize(tZ, tnu_Z, nb_iter = nb_iter1, flag = 'all')
    print("sum tnnu_Z after", tnnu_Z.sum())

    nZ = tnZ.detach().cpu().numpy()
    nnu_Z = tnnu_Z.detach().cpu().numpy()
    
    # determine amount by which particles have moved and by which distributions have changed (squared error distance)
    distMove = np.sqrt(np.sum((Z - nZ)**2,axis=-1))
    distAlt = np.sqrt(np.sum((nu_Z-nnu_Z)**2,axis=-1))
    fig,ax = plt.subplots(2,1)
    ax[0].hist(distMove)
    ax[1].hist(distAlt)
    ax[0].set_title('Distance Moved')
    ax[1].set_title('Distribution Changed')
    fig.savefig(outpath+'_optimalZnu_ZwC' + str(C) + '_sig' + str(sig) + '_Nmax' + str(Nmax) + '_Npart' + str(Npart) + '_distances.png',dpi=300)
    np.savez(outpath+'_optimalZnu_ZwC' + str(C) + '_sig' + str(sig) + '_Nmax' + str(Nmax) + '_Npart' + str(Npart) + '_distances.npz',Z=Z,nZ=nZ,nu_Z=nu_Z,nnu_Z=nnu_Z,distMove=distMove,distAlt=distAlt)

    # Not good !!!! -- removes low mass particles 
    mean = np.mean(nnu_Z.sum(axis=1))
    c = nnu_Z.sum(axis=1)>0.1*mean
    print("number of orig particles " + str(nnu_Z.shape[0]))
    print("number of new particles after remove low mass " + str(np.sum(c)))
    nZ = nZ[c,:]
    nnu_Z = nnu_Z[c,:]

    # Display of the result (in reduced integer set)
    maxInd = np.argmax(nnu_Z,axis=-1)+1
    vtf.writeVTK(nZ,[maxInd,np.sum(nnu_Z,axis=-1)],['MAX_VAL_NU','TOTAL_MASS'],outpath+'_optimalZnu_ZwC' + str(C) + '_sig' + str(sig) + '_Nmax' + str(Nmax) + '_Npart' + str(Npart) +'.vtk',polyData=None)
    np.savez(outpath+'_optimalZnu_ZwC' + str(C) + '_sig' + str(sig) + '_Nmax' + str(Nmax) + '_Npart' + str(Npart) +'.npz',Z=nZ, nu_Z=nnu_Z)

    '''
    # remove to see if release memory 
    f,ax = plt.subplots()
    ax.plot(np.arange(len(lossTrack)),np.asarray(lossTrack))
    ax.set_xlabel('iterations')
    ax.set_ylabel('cost')
    f.savefig(outpath+ '_optimalZcost_sig' + str(sig) + '_C' + str(C) + '_Nmax' + str(Nmax) + '_Npart' + str(Npart) + '.png',dpi=300)
    '''
    
    return nZ, nnu_Z