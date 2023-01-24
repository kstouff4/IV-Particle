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

def getXZ(npzX,npzZ):
    '''
    Assume initial Z is already in one hot encoding for features
    '''
    xInfo = np.load(npzX)
    X = xInfo['X']
    zInfo = np.load(npzZ)
    Z = zInfo['Z']
    nu_X, nu_Z,indsToKeep = oneHot(xInfo['nu_X'],zInfo['nu_Z'])

    return X,nu_X,Z,nu_Z,indsToKeep

def returnToInt(oneHotEncode):
    d = oneHotEncode.shape[-1]
    x = np.argmax(oneHotEncode,axis=-1)
    return x[...,None],d

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
                        

def project3D(X,nu_X,eps, sigma, nb_iter0, nb_iter1,outpath,Z=None,nu_Z=None,oneHot=True):
    '''
    Find and optimize subsample of points for X,nu_X defined by volume of X and sigma
    
    eps = denotes size of blocks for block sparse reduction
    sigma = sigma of space kernel 
    nb_iter0 = (2)
    nb_iter1 = (30)
    outpath = where to save results 
    Z, nu_Z = if already computed Z's
    '''
    C=1.2
    sig = sigma
    lossTrack = []
    
    process = psutil.Process(os.getpid())
    print("memory (bytes)")
    print(process.memory_info().rss)  # in bytes

    def make_subsample(X, nu_X,  N):
        # Input:
        # X is the initial data
        # N is the number of random subsample

        # Output 
        # Y is the output subsample

        sub_ind = np.random.choice(X.shape[0],replace = False, size = N)
        print(sub_ind.shape)
        Z = X[sub_ind,:]
        nu_Z = nu_X[sub_ind,:]*X.shape[0]/N
        return Z, nu_Z
    
    def make_ranges(X,Z,eps, sig):
      # Here X and Z are torch tensors

        X_labels = grid_cluster(X, eps) 
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

        Z_labels = grid_cluster(Z, eps) 
        Z_ranges, Z_centroids, _ = cluster_ranges_centroids(Z, Z_labels)
        D = ((Z_centroids[:, None, :] - Z_centroids[None, :, :]) ** 2).sum(dim=2)
        keep = D <(a*eps+4* sig) ** 2
        rangesZZ_ij = from_matrix(Z_ranges, Z_ranges, keep)
        areas = (Z_ranges[:, 1] - Z_ranges[:, 0])[:, None] * (Z_ranges[:, 1] 
                        - Z_ranges[:, 0])[None, :]
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
        print("memory (bytes)")
        print(process.memory_info().rss)  # in bytes
        return [rangesXX_ij, rangesZZ_ij, rangesZX_ij], [X_labels, Z_labels]

    def make_loss(tX, tnu_X, len_Z, dim_nu_Z, ranges):
        rangesXX_ij, rangesZZ_ij, rangesZX_ij = ranges # Getting the ranges
        LX_i = LazyTensor(tX[:,None,:])
        LX_j = LazyTensor(tX[None,:,:])
        
        print("dim nu Z")
        print(dim_nu_Z)
        if (oneHot):
            Lnu_X_i = LazyTensor(tnu_X[:,None,...].type(dtype)).one_hot(dim_nu_Z)
            Lnu_X_j = LazyTensor(tnu_X[None,:,...].type(dtype)).one_hot(dim_nu_Z)
        else:
            Lnu_X_i = LazyTensor(tnu_X[:,None,...].type(dtype))
            Lnu_X_j = LazyTensor(tnu_X[None,:,...].type(dtype))

        D_ij = ((LX_i - LX_j)**2/sig**2).sum(dim=2)  
        K_ij = (- D_ij).exp()
        P_ij = (Lnu_X_i*Lnu_X_j).sum(dim=2)
        KP_ij = K_ij*P_ij
        KP_ij.ranges = rangesXX_ij
        c=  KP_ij.sum(dim=1).sum()
        print('c=',c)

        def loss(tZal_Z):
            LZ_i, LZ_j = Vi(tZal_Z[0:3*len_Z].view(-1,3)), Vj(tZal_Z[0:3*len_Z].view(-1,3))
    
            Lnu_Z_i= Vi(tZal_Z[3*len_Z::].view(-1,dim_nu_Z)**2)
            Lnu_Z_j= Vj(tZal_Z[3*len_Z::].view(-1,dim_nu_Z)**2)

            DZZ_ij = ((LZ_i - LZ_j)**2/sig**2).sum(dim=2)  
            KZZ_ij = (- DZZ_ij).exp()  
            PZZ_ij = (Lnu_Z_i*Lnu_Z_j).sum(dim=2)
            KPZZ_ij = KZZ_ij*PZZ_ij
            KPZZ_ij.ranges = rangesZZ_ij

            DZX_ij = ((LZ_i - LX_j)**2/sig**2).sum(dim=2) 
            KZX_ij = (- DZX_ij).exp() 
            PZX_ij = (Lnu_Z_i*Lnu_X_j).sum(dim=2)
            KPZX_ij = KZX_ij*PZX_ij
            KPZX_ij.ranges = rangesZX_ij

            E = KPZZ_ij.sum(dim=1)-2*KPZX_ij.sum(dim=1)
            L = E.sum() +c
            return L
        return loss 
    
    def make_loss2(tX, tnu_X, tZ, dim_nu_Z, ranges):
        rangesXX_ij, rangesZZ_ij, rangesZX_ij = ranges # Getting the ranges
        print("Computing loss2")

        LX_i = LazyTensor(tX[:,None,:])
        LX_j = LazyTensor(tX[None,:,:])

        print("dim nu Z")
        print(dim_nu_Z)
        
        #Lnu_X_i = LazyTensor(tnu_X.type(dtype)).one_hot(dim_nu_Z)
        if (oneHot):
            Lnu_X_i = LazyTensor(tnu_X[:,None,...].type(dtype)).one_hot(dim_nu_Z)
            Lnu_X_j = LazyTensor(tnu_X[None,:,...].type(dtype)).one_hot(dim_nu_Z)
        else:
            Lnu_X_i = LazyTensor(tnu_X[:,None,...].type(dtype))
            Lnu_X_j = LazyTensor(tnu_X[None,:,...].type(dtype))
        
        D_ij = ((LX_i - LX_j)**2/sig**2).sum(dim=2)  
        K_ij = (- D_ij).exp()
        P_ij = (Lnu_X_i*Lnu_X_j).sum(dim=2) # P_ij = (Lnu_X_i == Lnu_X_j)
        KP_ij = K_ij*P_ij
        KP_ij.ranges = rangesXX_ij
        c=  KP_ij.sum(dim=1).sum()
        print('c=',c)

        LZ_i, LZ_j= Vi(tZ), Vj(tZ)
        DZZ_ij = ((LZ_i - LZ_j)**2/sig**2).sum(dim=2)  
        KZZ_ij = (- DZZ_ij).exp()    

        def loss(tal_Z):   
            Lnu_Z_i, Lnu_Z_j = Vi(tal_Z.view(-1,dim_nu_Z)**2), Vj(tal_Z.view(-1,dim_nu_Z)**2)

            PZZ_ij = (Lnu_Z_i*Lnu_Z_j).sum(dim=2)
            KPZZ_ij = KZZ_ij*PZZ_ij
            KPZZ_ij.ranges = rangesZZ_ij

            DZX_ij = ((LZ_i - LX_j)**2/sig**2).sum(dim=2) 
            KZX_ij = (- DZX_ij).exp() 
            PZX_ij = (Lnu_Z_i*Lnu_X_j).sum(dim=2)
            KPZX_ij = KZX_ij*PZX_ij
            KPZX_ij.ranges = rangesZX_ij
      
            E = KPZZ_ij.sum(dim=1)-2*KPZX_ij.sum(dim=1)
            L = E.sum() +c
            return L
        print("memory (bytes)")
        print(process.memory_info().rss)  # in bytes

        return loss

    def make_loss3(X, n_X, BZ, Bnu_Z):
        '''
        Includes boundary terms we are making loss with
        Note (11/14/22): not optimized for block sparse reduction yet 
        '''
  
        tx = torch.tensor(X).type(dtype).contiguous()
        tBz = torch.tensor(BZ).type(dtype).contiguous()
 
        LX_i, LX_j= Vi(tx), Vj(tx)
        LBZ_i, LBZ_j= Vi(tBz), Vj(tBz)

        tnu_X = torch.tensor(nu_X).type(dtype).contiguous()
        tBnu_Z = torch.tensor(Bnu_Z).type(dtype).contiguous()

        Lnu_X_i, Lnu_X_j = Vi(tnu_X), Vj(tnu_X)
        LBnu_Z_i, LBnu_Z_j = Vi(tBnu_Z), Vj(tBnu_Z)

        D_ij = ((LX_i - LX_j)**2/sig**2).sum(dim=2)  
        K_ij = (- D_ij).exp()    
        P_ij = (Lnu_X_i*Lnu_X_j).sum(dim=2)


        DBZ_ij = ((LBZ_i - LBZ_j)**2/sig**2).sum(dim=2)  
        KBZ_ij = (- DBZ_ij).exp()    
        PBZ_ij = (LBnu_Z_i*LBnu_Z_j).sum(dim=2)

        DBZX_ij = ((LBZ_i - LX_j)**2/sig**2).sum(dim=2)  
        KBZX_ij = (- DBZX_ij).exp()    
        PBZX_ij = (LBnu_Z_i*Lnu_X_j).sum(dim=2)

        c= (K_ij*P_ij).sum(dim=1).sum() + (KBZ_ij*PBZ_ij).sum(dim=1).sum() -2*(KBZX_ij*PBZX_ij).sum(dim=1).sum()
        print('c=',c)

        def loss(tZal_Z):
            LZ_i, LZ_j = Vi(tZal_Z[:,0:3].contiguous()), Vj(tZal_Z[:,0:3].contiguous())
      
            Lnu_Z_i= Vi(tZal_Z[:,3::].contiguous()**2)
            Lnu_Z_j= Vj(tZal_Z[:,3::].contiguous()**2)

            DZZ_ij = ((LZ_i - LZ_j)**2/sig**2).sum(dim=2)  
            KZZ_ij = (- DZZ_ij).exp()    
            PZZ_ij = (Lnu_Z_i*Lnu_Z_j).sum(dim=2)

            DZX_ij = ((LZ_i - LX_j)**2/sig**2).sum(dim=2) 
            KZX_ij = (- DZX_ij).exp() 
            PZX_ij = (Lnu_Z_i*Lnu_X_j).sum(dim=2)

            DZBZ_ij = ((LZ_i - LBZ_j)**2/sig**2).sum(dim=2) 
            KZBZ_ij = (- DZBZ_ij).exp() 
            PZBZ_ij = (Lnu_Z_i*Lnu_BZ_j).sum(dim=2)
        
            E = (KZZ_ij*PZZ_ij).sum(dim=1) -2*(KZX_ij*PZX_ij).sum(dim=1) + 2*(KZBZ_ij*PZBZ_ij).sum(dim=1) 
            L = E.sum() +c
            return L
        return loss 

    VBB = np.prod(np.max(X,axis=0)[0:3]-np.min(X,axis=0)[0:3]) # Volume of the bounding box

    sigma_min = (C*VBB/X.shape[0])**(1/3)
    print('sigma_min', sigma_min)
    if sigma < sigma_min:
        print('sigma is to small')
        print('min sigma:', sigma_min)
        exit

  # Random sampling   
    N = int(VBB/(C*sigma)**3)
    if (Z is None):
        Z, nu_Z = make_subsample(X, nu_X, N)

    tX = torch.tensor(X).type(dtype)
    if (oneHot):
        tnu_X = torch.tensor(nu_X).type(dtype)
    else:
        tnu_X = torch.tensor(nu_X).type(torch.bool)
    tZ = torch.tensor(Z).type(dtype)
    tnu_Z = torch.tensor(nu_Z).type(dtype)
    
    # Computes ranges and labels for the grid
    print("Making ranges")
    ranges, labels = make_ranges(tX,tZ, eps, sig)
    X_labels, Z_labels = labels

    # Sorts X and nu_X
    print("Sorting X and nu_X")
    tX, _ = sort_clusters(tX, X_labels) # sorting the labels
    tnu_X, _ = sort_clusters(tnu_X, X_labels)

    #  Sorts Z and nu_Z
    print("Sorting Z and nu_Z")
    tZ, _ = sort_clusters(tZ, Z_labels) # sorting the labels
    tnu_Z, _ = sort_clusters(tnu_Z, Z_labels)
    len_Z, dim_nu_Z = nu_Z.shape

    # Optimization

    def optimize(tZ, tnu_Z, nb_iter = 20, flag = 'all'):
        if flag == 'all':
            loss = make_loss(tX, tnu_X,len_Z, dim_nu_Z, ranges)
            p0 = torch.cat((tZ.flatten(),tnu_Z.pow(0.5).flatten()),0).requires_grad_(True)
        else:
            loss = make_loss2(tX, tnu_X, tZ, dim_nu_Z, ranges)
            p0 = tnu_Z.pow(0.5).flatten().clone().requires_grad_(True)

        print('p0', p0.is_contiguous())
        optimizer = torch.optim.LBFGS([p0], max_eval=10, max_iter=100, line_search_fn = 'strong_wolfe')
    
        def closure():
            optimizer.zero_grad()
            L = loss(p0)
            print("loss", L.detach().cpu().numpy())
            lossTrack.append(L.detach().cpu().numpy())
            L.backward()
            return L

        for i in range(nb_iter):
            print("it ", i, ": ", end="")
            optimizer.step(closure)

        if flag == 'all':
            tnZ = p0[0:3*len_Z].detach().view(-1,3)
            tnnu_Z = p0[3*len_Z::].detach().view(-1,dim_nu_Z)**2
        else:
            tnZ = tZ
            tnnu_Z = p0.detach().view(-1,dim_nu_Z)**2
        return tnZ, tnnu_Z

    print("Starting Optim")
    print("sum tnu_X", tnu_X.shape[0])
    print("sum tnu_Z before", tnu_Z.sum())
    tZ, tnu_Z = optimize(tZ, tnu_Z, nb_iter = nb_iter0, flag = '')
    tnZ, tnnu_Z = tZ, tnu_Z
    tnZ, tnnu_Z = optimize(tZ, tnu_Z, nb_iter = nb_iter1, flag = 'all')
    print("sum tnnu_Z after", tnnu_Z.sum())

    nZ = tnZ.detach().cpu().numpy()
    nnu_Z = tnnu_Z.detach().cpu().numpy()
    #nZ = nZ.reshape(len(nZ)/3,3)
    #nnu_Z = nnu_Z.reshape(len(nZ)/3, 3*len(nnu_Z)/len(nZ))

    # Not good !!!! -- removes low mass particles 
    mean = np.mean(nnu_Z.sum(axis=1))
    c = nnu_Z.sum(axis=1)>0.1*mean
    nZ = nZ[c,:]
    nnu_Z = nnu_Z[c,:]

    # Display of the result (in reduced integer set)
    maxInd = np.argmax(nnu_Z,axis=-1)+1
    vtf.writeVTK(nZ,[maxInd,np.sum(nnu_Z,axis=-1)],['MAX_VAL_NU','TOTAL_MASS'],outpath+'_optimalZnu_ZwC' + str(C) + '_sig' + str(sig) + 'eps' + str(eps) + '.vtk',polyData=None)
    np.savez(outpath+'_optimalZnu_ZwC' + str(C) + '_sig' + str(sig) + '.npz',Z=nZ, nu_Z=nnu_Z)
                            
    f,ax = plt.subplots()
    ax.plot(np.arange(len(lossTrack)),np.asarray(lossTrack))
    ax.set_xlabel('iterations')
    ax.set_ylabel('cost')
    f.savefig(outpath+ '_optimalZcost_sig' + str(sig) + '_C' + str(C) + 'eps' + str(eps) +'.png',dpi=300)
    
    return nZ, nnu_Z