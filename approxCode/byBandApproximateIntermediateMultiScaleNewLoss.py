import time
import sys
from sys import path as sys_path
sys_path.append('/cis/home/kstouff4/Documents/SurfaceTools/')
import vtkFunctions as vtf

import torch
import numpy as np
import numpy.matlib
import multiprocessing as mp
from multiprocessing import Pool

import os, psutil
from matplotlib import pyplot as plt
import matplotlib
if 'DISPLAY' in os.environ:
    matplotlib.use('qt5Agg')
else:
    matplotlib.use("Agg")

import pykeops
from pykeops.torch import LazyTensor

from pykeops.torch import Vi, Vj

from pykeops.torch.cluster import sort_clusters
from pykeops.torch.cluster import cluster_ranges_centroids
#from pykeops.torch.cluster import grid_cluster
from pykeops.torch.cluster import from_matrix

np_dtype = "float32"
dtype = torch.cuda.FloatTensor 
torch.set_default_tensor_type(dtype)

#pykeops.clean_pykeops()
plt.ion()

#torch.cuda.empty_cache()
import GPUtil
GPUtil.showUtilization()

import pdb
################################################################################
# grid Cluster New
def grid_cluster(x, size):
    r"""Simplistic clustering algorithm which distributes points into cubic bins.

    Args:
        x ((M,D) Tensor): List of points :math:`x_i \in \mathbb{R}^D`.
        size (float or (D,) Tensor): Dimensions of the cubic cells ("voxels").

    Returns:
        (M,) IntTensor:

        Vector of integer **labels**. Two points ``x[i]`` and ``x[j]`` are
        in the same cluster if and only if ``labels[i] == labels[j]``.
        Labels are sorted in a compact range :math:`[0,C)`,
        where :math:`C` is the number of non-empty cubic cells.

    Example:
        >>> x = torch.Tensor([ [0.], [.1], [.9], [.05], [.5] ])  # points in the unit interval
        >>> labels = grid_cluster(x, .2)  # bins of size .2
        >>> print( labels )
        tensor([0, 0, 2, 0, 1], dtype=torch.int32)

    """
    print("using my grid cluster")
    with torch.no_grad():
        # Quantize the points' positions
        if x.shape[1] == 1:
            weights = torch.IntTensor(
                [1],
            ).to(x.device)
        elif x.shape[1] == 2:
            weights = torch.IntTensor(
                [2**10, 1],
            ).to(x.device)
        elif x.shape[1] == 3:
            weights = torch.IntTensor([2**20, 2**10, 1]).to(x.device)
        else:
            raise NotImplementedError()
        x_ = ((x-x.min(axis=0,keepdim=True).values) / size).floor().int()
        print("number of unique cubes in x,y,z")
        print(len(torch.unique(x_[:,0])))
        print(len(torch.unique(x_[:,1])))
        print(len(torch.unique(x_[:,2])))
        qt = x.max(axis=0,keepdim=True).values - x.min(axis=0,keepdim=True).values
        print("with ranges, ", qt)
        vol = qt[0]
        for j in range(1,len(qt)):
            if (qt[j] > 0):
                vol *= qt[j]
        print("and volume, ", vol)
              
        x_ *= weights
        lab = x_.sum(1)  # labels
        lab = lab - lab.min()

        # Replace arbitrary labels with unique identifiers in a compact arange
        u_lab = torch.unique(lab).sort()[0]
        N_lab = len(u_lab)
        foo = torch.empty(u_lab.max() + 1, dtype=torch.int32, device=x.device)
        foo[u_lab] = torch.arange(N_lab, dtype=torch.int32, device=x.device)
        lab = foo[lab]

    return lab

# Helper Functions

def reverseOneHot(nu_Z,indsToKeep,maxVal=673):
    '''
    assume IndsToKeep indicate the mapping of original labels
    '''
    nnuZ = np.zeros((nu_Z.shape[0],maxVal))
    nnuZ[:,indsToKeep] = nu_Z
    return nnuZ

def oneHotMemorySave(nu_X,nu_Z,zeroBased=False):
    nonZeros = np.sum(nu_Z,axis=0) # values with 0 have no counts
    x = np.unique(nu_X).astype(int)
    if not zeroBased:
        x = x-1
        print("subtracting")
    nonZeros[x] += 1
    indsToKeep = np.where(nonZeros > 0)
    nnu_Z = nu_Z[:,indsToKeep[0]]
    return nnu_Z,indsToKeep[0]

def groupXbyLabelSave(nuX,X,indsToKeep,zeroBased=False):
    d = len(indsToKeep) # Z has all of possible labels
    listOfX = []
    iTK = indsToKeep
    if (not zeroBased):
        iTK += 1
    for i in iTK:
        listOfX.append(X[np.squeeze(nuX == i),...])
    return listOfX

def getXinterZ(npzX,npzZ):
    '''
    Assume initial Z is already in one hot encoding for features
    '''
    xInfo = np.load(npzX)
    X = xInfo[xInfo.files[0]]
    nuX = xInfo[xInfo.files[1]] # assume both have same features
    
    if len(np.squeeze(nuX).shape) == 1:
        if np.min(nuX) > 0:
            nuX = nuX - 1
        nuXn = np.zeros((X.shape[0],np.max(nuX)+1))
        nuXn[np.arange(nuX.shape[0]),nuX] = 1.0
        nuX = nuXn

    zInfo = np.load(npzZ)
    Z = zInfo['Z']
    nuZ = zInfo['nu_Z']
    
    print("shapes of features should be the same")
    print(nuX.shape)
    print(nuZ.shape)
    print("min and max of nuZ")
    print(np.min(nuZ,axis=0))
    print(np.max(nuZ,axis=0))
    
    print("sums of nu should be the same")
    print(np.sum(nuX))
    print(np.sum(nuZ))
   
    print(nuX)
    print(nuZ)
    
    return X,nuX,Z,nuZ

def rescale_loss(loss,lossOnly,x,dmax):
    # implement a change of variable of the loss for lbfg
    # x and dmax as suppose to be a list of tensors for each group of variables
    print("maximum changes, ")
    for d in dmax:
        print(d.detach().cpu().numpy())
    xr = []
    # Create a randomized initialization point
    for (xh,dmaxh) in zip(x,dmax):
        xr.append((torch.clone(xh).detach()+(2*torch.rand(xh.shape).type(dtype)-1)*dmaxh).requires_grad_(True))

    # Compute the gradient at randomized intial point
    L = lossOnly(xr)
    L.backward()

    # Compute the scaling coefficient
    a, cp = [], 0.
    for (xrh,dmaxh) in zip(xr,dmax):
        ah = (dmaxh/(xrh.grad.max())).sqrt()
        cp += ah*(xrh.grad.abs().sum())
        a.append(ah)
    cp = torch.max(torch.tensor(1.),cp)
    print("cp is, ", cp.detach().cpu().numpy())
    
    print("xr is")
    print(len(xr))
    for ii in xr:
        print(ii.detach().shape)

    a = [ah*cp for ah in a]

    def uvar_from_optvar(tx):
        # tranform optimization var into user variable
        return [(xh*ah) for (xh,ah) in zip(tx,a)]

    def loss_new(xopt_cur):
        # create the new rescaled using the optimization variable
        return loss(uvar_from_optvar(xopt_cur))

    xopt = []
    for (xh,ah) in zip(x,a):
        xopt.append(torch.clone(xh/ah).detach().requires_grad_(True))
        print("ah:", ah.detach())

    return loss_new, uvar_from_optvar, xopt 

#################################################################################
# Main Function
def project3D(Xfile, sigma, nb_iter0, nb_iter1,outpath,Nmax=2000.0,Npart=50000.0,Zfile=None,maxV=673,optMethod='LBFGS',C=1.2,bw=70):
    '''
    Find and optimize subsample of points for X,nu_X defined by volume of X and sigma
    
    eps = denotes size of blocks for block sparse reduction
    sigma = list of sigmas with last one = largest and dictating the ranges 
    nb_iter0 = (2)
    nb_iter1 = (30)
    outpath = where to save results 
    Z, nu_Z = if already computed Z's
    
    Yongsoo max = 351
    Allen max = 673
    
    OptMethods = LBFGS or Adam (less memory intensive)
    
    bw = number of features to consider at one time (harnesses orthogonality of features, not feasible if feature dimensions not independent)
    '''
    st = time.time()
    #C=1.2
    sig = sigma
    print(sig)
    lossTrack = []
    
    process = psutil.Process(os.getpid())

    def make_ranges(X, Z, epsX, epsZ,sigP):
      # Here X and Z are torch tensors
        a = np.sqrt(3)
        Z_labels = grid_cluster(Z, epsZ) 
        Z_ranges, Z_centroids, _ = cluster_ranges_centroids(Z, Z_labels)
        #D = ((LazyTensor(Z_centroids[:, None, :]) - LazyTensor(Z_centroids[None, :, :])) ** 2).sum(dim=2)
        D = ((Z_centroids[:,None,:] - Z_centroids[None,:,:])**2).sum(dim=2)
        keep = D <(a*epsZ+4* sigP) ** 2
        rangesZZ_ij = from_matrix(Z_ranges, Z_ranges, keep)
        print(rangesZZ_ij[2].shape)
        areas = (Z_ranges[:, 1] - Z_ranges[:, 0])[:, None] * (Z_ranges[:, 1] 
                        - Z_ranges[:, 0])[None, :]
        total_area = areas.sum()  # should be equal to N*M
        sparse_area = areas[keep].sum()
        print(
        "We keep {:.2e}/{:.2e} = {:2d}% of the original kernel matrix.".format(
          sparse_area, total_area, int(100 * sparse_area / total_area)))
        print("")
        
        X_labels = grid_cluster(X,epsX)
        X_ranges, X_centroids, _ = cluster_ranges_centroids(X,X_labels)
        D = ((X_centroids[:,None,:] - X_centroids[None,:,:])**2).sum(dim=2)
        keep = D <(a*epsX+4* sigP) ** 2
        rangesXX_ij = from_matrix(X_ranges, X_ranges, keep)
        print(rangesXX_ij[2].shape)
        areas = (X_ranges[:, 1] - X_ranges[:, 0])[:, None] * (X_ranges[:, 1] 
                        - X_ranges[:, 0])[None, :]
        total_area = areas.sum()  # should be equal to N*M
        sparse_area = areas[keep].sum()
        print(
        "We keep {:.2e}/{:.2e} = {:2d}% of the original kernel matrix.".format(
          sparse_area, total_area, int(100 * sparse_area / total_area)))
        print("")
        
        D = ((Z_centroids[:, None, :] - X_centroids[None, :, :]) ** 2).sum(dim=2)
        keep = D < (a*(epsZ/2.0 + epsX/2.0)+4*sigP)**2
        rangesZX_ij = from_matrix(Z_ranges, X_ranges, keep)
        areas = (Z_ranges[:, 1] - Z_ranges[:, 0])[:, None] * (X_ranges[:, 1] 
                            - X_ranges[:, 0])[None, :]
        total_area = areas.sum()  # should be equal to N*M
        sparse_area = areas[keep].sum()
        
        return rangesXX_ij, rangesZZ_ij, rangesZX_ij, X_labels, Z_labels
    
    def getInitialLoss(tX,tnuX,tZ,tnuZ,rangesXX_ij,rangesZZ_ij,rangesZX_ij):
        c = 0
        LX_i = LazyTensor(tX[:,None,:])
        LX_j = LazyTensor(tX[None,:,:])
        
        Lnu_X_i = Vi(tnuX)
        Lnu_X_j = Vj(tnuX)
        PXX_ij = (Lnu_X_i*Lnu_X_j).sum(dim=2)
        for si in range(len(sig)):
            sigg = torch.tensor(sig[si]).type(dtype)
            D_ij = ((LX_i - LX_j)**2/sigg**2).sum(dim=2) 
            K_ijs = (- D_ij).exp()
            if si == 0:
                K_ij = K_ijs
            else:
                K_ij += K_ijs
                
        K_ij = K_ij*PXX_ij
        K_ij.ranges = rangesXX_ij
        c +=  (K_ij.sum(dim=1).sum())
        
        LZ_i, LZ_j= Vi(tZ), Vj(tZ)
        for si in range(len(sig)):
            sigg = torch.tensor(sig[si]).type(dtype)
            DZZ_ij = ((LZ_i - LZ_j)**2/sigg**2).sum(dim=2)  
            KZZ_ijs = (- DZZ_ij).exp()
            if si == 0:
                KZZ_ij = KZZ_ijs
            else:
                KZZ_ij += KZZ_ijs
        
        Lnu_Z_i, Lnu_Z_j = Vi(tnuZ), Vj(tnuZ)
        PZZ_ij = (Lnu_Z_i*Lnu_Z_j).sum(dim=2)
        KPZZ_ij = KZZ_ij*PZZ_ij
        KPZZ_ij.ranges=rangesZZ_ij
        L = (KPZZ_ij.sum(dim=1).sum()) + c
            
        for si in range(len(sig)):
            sigg = torch.tensor(sig[si]).type(dtype)
            DZX_ij = ((LZ_i - LX_j)**2/sigg**2).sum(dim=2) 
            KZX_ijs = (- DZX_ij).exp() 
            if si == 0:
                KZX_ij = KZX_ijs
            else:
                KZX_ij += KZX_ijs

        PZX_ij = (Lnu_Z_i*Lnu_X_j).sum(dim=2)
        KPZX_ij = KZX_ij*PZX_ij
        KPZX_ij.ranges = rangesZX_ij
            
        L -= 2.0*KPZX_ij.sum(dim=1).sum()

        return L.detach()

    def make_loss(tX, tnuX, len_Z, dim_nu_Z, rangesXX_ij, rangesZZ_ij, rangesZX_ij,coeff=torch.tensor(1.0).type(dtype)):
        c = 0
        temptime = time.time()
        LX_i = LazyTensor(tX[:,None,:])
        LX_j = LazyTensor(tX[None,:,:])
        
        Lnu_X_i = Vi(tnuX)
        Lnu_X_j = Vj(tnuX)
        PXX_ij = (Lnu_X_i*Lnu_X_j).sum(dim=2)
        for si in range(len(sig)):
            sigg = torch.tensor(sig[si]).type(dtype)
            D_ij = ((LX_i - LX_j)**2/sigg**2).sum(dim=2) 
            K_ijs = (- D_ij).exp()
            if si == 0:
                K_ij = K_ijs
            else:
                K_ij += K_ijs
                
        K_ij = K_ij*PXX_ij
        K_ij.ranges = rangesXX_ij
        c +=  coeff*(K_ij.sum(dim=1).sum())
        print('c=',c.detach())

        def loss(tZal_Z):
            # z operation is still assumed to be the same (mixed distribution of labels)
            LZ_i, LZ_j = Vi(tZal_Z[0]), Vj(tZal_Z[0])
    
            Lnu_Z_i= Vi((tZal_Z[1])**2)
            Lnu_Z_j= Vj((tZal_Z[1])**2)
            for si in range(len(sig)):
                sigg = torch.tensor(sig[si]).type(dtype)
                DZZ_ij = ((LZ_i - LZ_j)**2/sigg**2).sum(dim=2)  
                KZZ_ijs = (- DZZ_ij).exp()  
                if si == 0:
                    KZZ_ij = KZZ_ijs
                else:
                    KZZ_ij += KZZ_ijs
            PZZ_ij = (Lnu_Z_i*Lnu_Z_j).sum(dim=2)
            KPZZ_ij = KZZ_ij*PZZ_ij
            KPZZ_ij.ranges = rangesZZ_ij
            
            for si in range(len(sig)):
                sigg = torch.tensor(sig[si]).type(dtype)
                DZX_ij = ((LZ_i - LX_j)**2/sigg**2).sum(dim=2)
                KZX_ijs = (- DZX_ij).exp()
                if si == 0:
                    KZX_ij = KZX_ijs
                else:
                    KZX_ij += KZX_ijs
            
            PZX_ij = (Lnu_Z_i*Lnu_X_j).sum(dim=2)
            KPZX_ij = KZX_ij*PZX_ij
            KPZX_ij.ranges = rangesZX_ij
            
            L = KPZZ_ij.sum(dim=1).sum() - 2.0*KPZX_ij.sum(dim=1).sum()
            L = L*coeff
            L.backward()
            L += c
            return L.detach(),c.detach()
    
        def lossOnly(tZal_Z):
            # z operation is still assumed to be the same (mixed distribution of labels)
            LZ_i, LZ_j = Vi(tZal_Z[0]), Vj(tZal_Z[0])
    
            Lnu_Z_i= Vi((tZal_Z[1])**2)
            Lnu_Z_j= Vj((tZal_Z[1])**2)
            for si in range(len(sig)):
                sigg = torch.tensor(sig[si]).type(dtype)
                DZZ_ij = ((LZ_i - LZ_j)**2/sigg**2).sum(dim=2)  
                KZZ_ijs = (- DZZ_ij).exp()  
                if si == 0:
                    KZZ_ij = KZZ_ijs
                else:
                    KZZ_ij += KZZ_ijs
            PZZ_ij = (Lnu_Z_i*Lnu_Z_j).sum(dim=2)
            KPZZ_ij = KZZ_ij*PZZ_ij
            KPZZ_ij.ranges = rangesZZ_ij
            
            for si in range(len(sig)):
                sigg = torch.tensor(sig[si]).type(dtype)
                DZX_ij = ((LZ_i - LX_j)**2/sigg**2).sum(dim=2)
                KZX_ijs = (- DZX_ij).exp()
                if si == 0:
                    KZX_ij = KZX_ijs
                else:
                    KZX_ij += KZX_ijs
            
            PZX_ij = (Lnu_Z_i*Lnu_X_j).sum(dim=2)
            KPZX_ij = KZX_ij*PZX_ij
            KPZX_ij.ranges = rangesZX_ij
            
            L = KPZZ_ij.sum(dim=1).sum() - 2.0*KPZX_ij.sum(dim=1).sum()
            L = L*coeff
            L += c
            return L
        return loss,lossOnly
    
    def make_loss2(tX, tnuX, tZ, dim_nu_Z, rangesXX_ij, rangesZZ_ij, rangesZX_ij,coeff=torch.tensor(1.0).type(dtype)):
        c = 0
        LX_i = LazyTensor(tX[:,None,:])
        LX_j = LazyTensor(tX[None,:,:])
        
        Lnu_X_i = Vi(tnuX)
        Lnu_X_j = Vj(tnuX)
        
        for si in range(len(sig)):
            sigg = torch.tensor(sig[si]).type(dtype)
            D_ij = ((LX_i - LX_j)**2/sigg**2).sum(dim=2) 
            K_ijs = (- D_ij).exp()
            if si == 0:
                K_ij = K_ijs
            else:
                K_ij += K_ijs
                
        PXX_ij = (Lnu_X_i*Lnu_X_j).sum(dim=2)
        K_ij = K_ij*PXX_ij
        K_ij.ranges = rangesXX_ij
        c +=  coeff*(K_ij.sum(dim=1).sum())
        print('c=',c.detach())

        LZ_i, LZ_j= Vi(tZ), Vj(tZ)
        for si in range(len(sig)):
            sigg = torch.tensor(sig[si]).type(dtype)
            DZZ_ij = ((LZ_i - LZ_j)**2/sigg**2).sum(dim=2)  
            KZZ_ijs = (- DZZ_ij).exp()
            if si == 0:
                KZZ_ij = KZZ_ijs
            else:
                KZZ_ij += KZZ_ijs
        
        def finalLoss(nuZ):
            tnuZ = torch.tensor(nuZ).type(dtype)
            Lnu_Z_i, Lnu_Z_j = Vi(tnuZ), Vj(tnuZ)
            PZZ_ij = (Lnu_Z_i*Lnu_Z_j).sum(dim=2)
            KPZZ_ij = KZZ_ij*PZZ_ij
            KPZZ_ij.ranges=rangesZZ_ij
            L = coeff*(KPZZ_ij.sum(dim=1).sum()) + c
            
            for si in range(len(sig)):
                sigg = torch.tensor(sig[si]).type(dtype)
                DZX_ij = ((LZ_i - LX_j)**2/sigg**2).sum(dim=2) 
                KZX_ijs = (- DZX_ij).exp() 
                if si == 0:
                    KZX_ij = KZX_ijs
                else:
                    KZX_ij += KZX_ijs

            PZX_ij = (Lnu_Z_i*Lnu_X_j).sum(dim=2)
            KPZX_ij = KZX_ij*PZX_ij
            KPZX_ij.ranges = rangesZX_ij
            
            L -= 2.0*coeff*KPZX_ij.sum(dim=1).sum()
            return L.detach(),c.detach()
        
        def lossOnly2(tal_Z):
            Lnu_Z_i, Lnu_Z_j = Vi(tal_Z[0]**2), Vj(tal_Z[0]**2)

            PZZ_ij = (Lnu_Z_i*Lnu_Z_j).sum(dim=2)
            KPZZ_ij = KZZ_ij*PZZ_ij
            KPZZ_ij.ranges = rangesZZ_ij
            
            for si in range(len(sig)):
                sigg = torch.tensor(sig[si]).type(dtype)
                DZX_ij = ((LZ_i - LX_j)**2/sigg**2).sum(dim=2) 
                KZX_ijs = (- DZX_ij).exp() 
                if si == 0:
                    KZX_ij = KZX_ijs
                else:
                    KZX_ij += KZX_ijs
            
            PZX_ij = (Lnu_Z_i*Lnu_X_j).sum(dim=2)
            KPZX_ij = KZX_ij*PZX_ij
            KPZX_ij.ranges = rangesZX_ij
            
            L = KPZZ_ij.sum(dim=1).sum() - 2.0*KPZX_ij.sum(dim=1).sum()
            L = coeff*L
            L += c
            return L

        def loss2(tal_Z): 
            Lnu_Z_i, Lnu_Z_j = Vi(tal_Z[0]**2), Vj(tal_Z[0]**2)

            PZZ_ij = (Lnu_Z_i*Lnu_Z_j).sum(dim=2)
            KPZZ_ij = KZZ_ij*PZZ_ij
            KPZZ_ij.ranges = rangesZZ_ij
            
            print("memory (bytes)")
            print(process.memory_info().rss)  # in bytes
            GPUtil.showUtilization()
            
            for si in range(len(sig)):
                sigg = torch.tensor(sig[si]).type(dtype)
                DZX_ij = ((LZ_i - LX_j)**2/sigg**2).sum(dim=2) 
                KZX_ijs = (- DZX_ij).exp() 
                if si == 0:
                    KZX_ij = KZX_ijs
                else:
                    KZX_ij += KZX_ijs
            
            PZX_ij = (Lnu_Z_i*Lnu_X_j).sum(dim=2)
            KPZX_ij = KZX_ij*PZX_ij
            KPZX_ij.ranges = rangesZX_ij
            
            L = KPZZ_ij.sum(dim=1).sum() - 2.0*KPZX_ij.sum(dim=1).sum()
            L = coeff*L
            L.backward()
            L += c
            return L.detach(),c.detach()

        print("memory (bytes)")
        print(process.memory_info().rss)  # in bytes
        return loss2, lossOnly2
    
    # get X and Z: requires input initial sample
    X,nu_X,Z,nu_Z = getXinterZ(Xfile,Zfile)
    # write original Z 
    maxInd = np.argmax(nu_Z,axis=-1)
    vtf.writeVTK(Z,[maxInd,np.sum(nu_Z,axis=-1)],['MAX_VAL_NU','TOTAL_MASS'],outpath+'_originalZnu_ZwC' + str(C) + '_sig' + str(sig) + '.vtk',polyData=None)
    
    # make Epislon List
    def makeEps(X,Z,Nmax,Npart):
        '''
        Returns list of epsilons to use for making ranges
        '''
        epsList = []
        volList = []
        partList = []
        ncubeList = []
        denZ = min(Z.shape[0]/Npart,Nmax)
        rangeOfData = np.max(Z,axis=0) - np.min(Z,axis=0)
        numDimNonzero = np.sum(rangeOfData > 0)
        if rangeOfData[-1] == 0:
            volZ = np.prod(rangeOfData[:-1])
            epsZ = (volZ/denZ)**(1.0/numDimNonzero)
        else: # assume 3D
            volZ = np.prod(np.max(Z,axis=0) - np.min(Z,axis=0)) # volume of bounding box  (avoid 0); 1 micron
            epsZ = np.cbrt(volZ/denZ)
        print("ZX\tVol\tParts\tCubes\tEps")
        print("Z\t" + str(volZ) + "\t" + str(Z.shape[0]) + "\t" + str(denZ) + "\t" + str(epsZ))

        den = min(X.shape[0]/Npart,Nmax)
        rangeOfData = np.max(X,axis=0) - np.min(X,axis=0)
        numDimNonzero = np.sum(rangeOfData > 0)
        if rangeOfData[-1] == 0:
            volX = np.prod(rangeOfData[:-1])
            epsX = (volX/den)**(1.0/numDimNonzero)
        else:
            volX = np.prod(np.max(X,axis=0)-np.min(X,axis=0))
            epsX = np.cbrt(volX/den)

        print("X\t" + str(volX) + "\t" + str(X.shape[0]) + "\t" + str(den) + "\t" + str(epsX))
        return epsX,epsZ
    
    temptime = time.time()
    epsX,epsZ = makeEps(X,Z,Nmax,Npart)
    print("time for making epsilon is " + str(time.time()-temptime)) 
    
    # make tensors for each
    tX = torch.tensor(X).type(dtype)
    tnu_X = torch.tensor(nu_X).type(dtype)
         
    tZ = torch.tensor(Z).type(dtype)
    tnu_Z = torch.tensor(nu_Z).type(dtype)
    

    # Computes ranges and labels for the grid
    print("Making ranges")
    temptime = time.time()
    rangesXX, rangesZZ, rangesZX, X_labels, Z_labels = make_ranges(tX, tZ, epsX, epsZ,sig[-1])
    print("time for making ranges is " + str(time.time() - temptime))
    
    # Sorts X and nu_X
    print("Sorting X and nu_X")
    temptime = time.time()
   
    #  Sorts Z and nu_Z
    print("Sorting Z and nu_Z")
    temptime = time.time()
    tZ, _ = sort_clusters(tZ, Z_labels) # sorting the labels
    tnu_Z, _ = sort_clusters(tnu_Z, Z_labels)
    tX,_ = sort_clusters(tX,X_labels)
    tnu_X, _ = sort_clusters(tnu_X,X_labels)
    print("time for sorting Z is " + str(time.time() - temptime))
    len_Z, dim_nu_Z = nu_Z.shape
    
    # if want to rescale loss such taht is equal to 1 (normalizing should be same with and without bands)
    coeff = getInitialLoss(tX,tnu_X,tZ,tnu_Z,rangesXX,rangesZZ,rangesZX)
    coeff = 1.0/coeff

    # Optimization
    outerCost = []
    nb_bands = int(nu_Z.shape[1]/bw)+1
    bands = [(i*bw, min((i+1)*bw,nu_Z.shape[1])) for i in range(nb_bands)] # define indices of bands in terms of feature dimensions
    def optimize(tZp, tnu_Zp, nb_iter = 20, flag = 'all',coeff=coeff):
        if flag == 'all':
            temptime = time.time()
            xinit = [torch.clone(tZp).detach(),torch.clone(tnu_Zp.pow(0.5)).detach()]
            # loss bands
            losses = []
            loss_onlys = []
            for b in range(nb_bands):
                loss,loss_only = make_loss(tX, tnu_X[:,bands[b][0]:bands[b][1]], len_Z, bands[b][1]-bands[b][0], rangesXX, rangesZZ, rangesZX,coeff=coeff)
                losses.append(loss)
                loss_onlys.append(loss_only)
            print("time for making loss is " + str(time.time() - temptime))
            
            def uloss(xu):
                return sum([losses[b]([xu[0],xu[1][:,bands[b][0]:bands[b][1]]]) for b in range(nb_bands)])
            def uloss_only(xu):
                return sum([loss_onlys[b]([xu[0],xu[1][:,bands[b][0]:bands[b][1]]]) for b in range(nb_bands)])
            # rescale the loss according to how much you are expecting the variables to change by computing the gradient with respect to the variables
            loss_new, uvar_from_optvar, xopt = rescale_loss(uloss,uloss_only,xinit,[torch.tensor(sig[0]).type(dtype),torch.clone(xinit[-1].mean()).detach().type(dtype)])
            optimizer = torch.optim.LBFGS(xopt, max_eval=10, max_iter=10, line_search_fn = 'strong_wolfe',history_size=5,tolerance_grad=1e-8,tolerance_change=1e-10)
        else:
            temptime = time.time()
            xinit = [torch.clone(tnu_Zp.pow(0.5)).detach()]
            # loss bands
            losses2 = []
            loss_onlys2 = []
            for b in range(nb_bands):
                loss2, loss_only2 = make_loss2(tX, tnu_X[:,bands[b][0]:bands[b][1]], tZp, bands[b][1]-bands[b][0], rangesXX, rangesZZ, rangesZX,coeff=coeff)
                losses2.append(loss2)
                loss_onlys2.append(loss_only2)
            print("time for making loss 2 is " + str(time.time() - temptime))
            def uloss2(xu):
                return sum([losses2[b]([xu[0][:,bands[b][0]:bands[b][1]]]) for b in range(nb_bands)])
            def uloss_only2(xu):
                return sum([loss_onlys2[b]([xu[0][:,bands[b][0]:bands[b][1]]]) for b in range(nb_bands)])
            loss_new2, uvar_from_optvar2, xopt2 = rescale_loss(uloss2,uloss_only2,xinit,[torch.clone(xinit[0].mean()).detach().type(dtype)])

            optimizer = torch.optim.LBFGS(xopt2, max_eval=10, max_iter=10, line_search_fn = 'strong_wolfe',history_size=5,tolerance_grad=1e-8,tolerance_change=1e-10)
        
        
        def closure():
            optimizer.zero_grad()
            if flag == 'all':
                L,c = loss_new(xopt)
                print("xopt")
                for o in xopt:
                    print(o.detach().cpu().numpy())
                xuser = uvar_from_optvar(xopt)
            else:
                L,c = loss_new2(xopt2)
                print("xopt2")
                for o in xopt2:
                    print(o.detach().cpu().numpy())
                xuser = uvar_from_optvar2(xopt2)
            
            print("error is ", L.detach().cpu().numpy())
            print("relative error loss", L.detach().cpu().numpy()/c.detach().cpu().numpy())
            lossTrack.append(L.detach().cpu().numpy()/c.detach().cpu().numpy())
            print("alpha: ")
            for o in xuser:
                print(o.detach().cpu().numpy())
            #tot.backward()
            return L

        for i in range(nb_iter):
            print("it ", i, ": ", end="")
            print(torch.cuda.memory_allocated(0))
            GPUtil.showUtilization()
            temptime = time.time()
            optimizer.step(closure)
            print("time to take a step is " + str(time.time() - temptime))
            osd = optimizer.state_dict()
            outerCost.append(np.copy(osd['state'][0]['prev_loss']))
            #torch.cuda.empty_cache()

        if flag == 'all':
            xuser = uvar_from_optvar(xopt)
            tnZ = xuser[0].detach()
            tnnu_Z = xuser[1].detach()**2
        else:
            tnZ = tZ
            xuser = uvar_from_optvar2(xopt2)
            tnnu_Z = xuser[0].detach()**2
        return tnZ, tnnu_Z

    print("Starting Optim")
    print("sum tnu_Z before", tnu_Z.detach().sum())
    print("starting tnu_z (user): ", tnu_Z.detach().cpu().numpy())
    ttZ, ttnu_Z = optimize(tZ, tnu_Z, nb_iter = nb_iter0, flag = '')
    tnZ, tnnu_Z = optimize(ttZ, ttnu_Z, nb_iter = nb_iter1, flag = 'all')
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
    
    fig,ax = plt.subplots()
    ax.plot(np.arange(len(outerCost)),outerCost)
    ax.set_title('Outer iterations Cost')
    ax.set_xlabel('Outer Step Iterations')
    ax.set_ylabel('Total Loss')
    fig.savefig(outpath+'_optimalZcost_sig' + str(sig) + '_C' + str(C) + '_Nmax' + str(Nmax) + '_Npart' + str(Npart) + '_OUTER_ONLY.png',dpi=300)
    
    bigMove = distMove > np.quantile(distMove,0.75)
    bigDist = distAlt > np.quantile(distAlt,0.75)
    
    vtf.writeVTK(Z[bigMove,:],[distMove[bigMove]],['SQUARED_DIST_MOVED'],outpath+'_optimalZmove_wC' + str(C) + '_sig' + str(sig) + '_Nmax' + str(Nmax) + '_Npart' + str(Npart) + '.vtk',polyData=None)
    vtf.writeVTK(Z[bigDist,:],[distAlt[bigDist]],['SQUARED_DIST_NU_ALTERED'],outpath+'_optimalnuZmove_wC' + str(C) + '_sig' + str(sig) + '_Nmax' + str(Nmax) + '_Npart' + str(Npart) + '.vtk',polyData=None)

    # Display of the result (in reduced integer set)
    maxInd = np.argmax(nnu_Z,axis=-1)
    nuZN = nnu_Z / (np.sum(nnu_Z,axis=-1)[...,None])
    h = nuZN*np.log(nuZN,where=(nuZN > 0))
    h = np.sum(h,axis=-1)
    h = -h
    
    geneInds = np.arange(nnu_Z.shape[-1]).astype(int)
    geneInds = list(geneInds)
    geneIndNames = []
    geneProbs = []
    for g in geneInds:
        geneIndNames.append(str(g))
        geneProbs.append(nnu_Z[:,g])
    vtf.writeVTK(nZ,[maxInd,np.sum(nnu_Z,axis=-1),h],['MAX_VAL_NU','TOTAL_MASS','ENTROPY'],outpath+'_optimalZnu_ZAllwC' + str(C) + '_sig' + str(sig) + '_Nmax' + str(Nmax) + '_Npart' + str(Npart) +'.vtk',polyData=None)
    vtf.writeVTK(nZ,geneProbs,geneIndNames,outpath+'_optimalZnu_ZAllwC' + str(C) + '_sig' + str(sig) + '_Nmax' + str(Nmax) + '_Npart' + str(Npart) +'_allComponents.vtk',polyData=None)
    np.savez(outpath+'_optimalZnu_ZAllwC' + str(C) + '_sig' + str(sig) + '_Nmax' + str(Nmax) + '_Npart' + str(Npart) +'.npz',Z=nZ, nu_Z=nnu_Z,h=h)
    

    # remove to see if release memory 
    f,ax = plt.subplots()
    ax.plot(np.arange(len(lossTrack)),np.asarray(lossTrack))
    ax.set_xlabel('iterations')
    ax.set_ylabel('cost')
    f.savefig(outpath+ '_optimalZcost_sig' + str(sig) + '_C' + str(C) + '_Nmax' + str(Nmax) + '_Npart' + str(Npart) + '.png',dpi=300)
    
    return nZ, nnu_Z
