import ntpath
from numba import jit, prange, int64 #, cuda
import time
import sys
from sys import path as sys_path
sys_path.append('/cis/home/kstouff4/Documents/SurfaceTools/')
import vtkFunctions as vtf

import multiprocessing as mp
from multiprocessing import Pool

import torch
import numpy as np
import numpy.matlib

import os, psutil
import matplotlib
from matplotlib import pyplot as plt
if 'DISPLAY' in os.environ:
    matplotlib.use('qt5Agg')
else:
    matplotlib.use("Agg")
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#import imageio.v3 as iio
import pykeops
import socket
pykeops.set_build_folder("~/.cache/keops"+pykeops.__version__ + "_" + (socket.gethostname()))

from pykeops.torch import LazyTensor
#import pykeops.config

from pykeops.torch import Vi, Vj

from pykeops.torch.cluster import sort_clusters
from pykeops.torch.cluster import cluster_ranges_centroids
from pykeops.torch.cluster import grid_cluster
from pykeops.torch.cluster import from_matrix

np_dtype = "float32"
dtype = torch.cuda.FloatTensor 


plt.ion()

#torch.cuda.empty_cache()
#cuda.current_context().deallocations.clear()
import GPUtil
GPUtil.showUtilization()

import glob
#import estimateSubsampleByLabelScratchTestExperiments as ess
##############################################################
'''
# Pseudocode

StitchQuadrants:
    For two quadrants, determine boundary in 0 or 1 direction:
        take difference of min and max; where difference is smallest, create band on either side of this, parallel with width 2*margin*sigma
        
    Combine datasets: X = X1 + X2, Zo = band, Z = Z1 + Z2 / Zo
    
    Make EPS and Make Ranges for eliminating computations (XZo, ZZo, ZoZo); only compute the elements that affect optimization
    
    Optimization:
        optimize nu_Zo only
        optimize nu_Zo + Zo
    
    Save combined datasets: X, Z + Zo following optimization

'''

def oneHot(nu_X,nu_Zopt,nu_Z):
    '''
    Make oneHot based on compilation of all of labels in X and Z's to optimize and not
    '''
    
    nnu_X = np.zeros((nu_X.shape[0],nu_Z.shape[-1])).astype('bool_') # assume nu_Z has full spectrum
    nnu_X[np.arange(nu_X.shape[0]),np.squeeze(nu_X-1).astype(int)] = 1
    
    nonZeros = np.sum(nnu_X,axis=0)+np.sum(nu_Z,axis=0)+np.sum(nu_Zopt,axis=0)
    indsToKeep = np.where(nonZeros > 0)
    print("total is " + str(len(indsToKeep[0]))) # 0 based with maximum = 1 less than dimension
    print(indsToKeep[0])
    
    nnu_X = nnu_X[:,indsToKeep[0]]
    nnu_Z = nu_Z[:,indsToKeep[0]]
    nnu_Zopt = nu_Zopt[:,indsToKeep[0]]

    return nnu_X,nnu_Zopt,nnu_Z,indsToKeep[0]

def reverseOneHot(nu_Zopt,indsToKeep,maxVal):    
    nnuZ = np.zeros((nu_Zopt.shape[0],maxVal))
    nnuZ[:,indsToKeep] = nu_Zopt
    return nnuZ

def groupXbyLabel(oneHotEncode,X):
    d = oneHotEncode.shape[-1]
    listOfX = []
    for i in range(d):
        listOfX.append(X[oneHotEncode[:,i]])
    return listOfX

def stitchQuadrants(x1,x2,z1,z2,outpath,sigma,nb_iter0, nb_iter1, margin=2.0,Nmax=2000.0,Npart=50000.0,maxV=673,optMethod='LBFGS'):
    '''
    Takes in 2 quadrants (reads in X and Z quadrants)
    Calculate min and max of first two axes
    if the difference of mins or maxes is <= 0.5, then assume next to each other and sharing that boundary
    define band based on boundary (1 or 2) 
    optimize particles using LBFGS by holding other parts of quadrant fixed
    save optimized 
    
    # assume the Z's have maxV 
    '''
    sig = sigma
    process = psutil.Process(os.getpid())
    lossTrack = []
    C=1.2
    
    print("working on X files")
    print(x1)
    print(x2)
    
    print("working on Z files")
    print(z1)
    print(z2)
    
    # First determine where boundaries are and whether there is a boundary to optimize 
    x1Info = np.load(x1)
    X1 = x1Info['X']
    x2Info = np.load(x2)
    X2 = x2Info['X']
    min1 = np.min(X1,axis=0)
    max1 = np.max(X1,axis=0)
    min2 = np.min(X2,axis=0)
    max2 = np.max(X2,axis=0)
    
    z1Info = np.load(z1)
    Z1 = z1Info['Z']
    nuZ1 = z1Info['nu_Z']
    
    z2Info = np.load(z2)
    Z2 = z2Info['Z']
    nuZ2 = z2Info['nu_Z']
    
    print(min1[0] - max2[0])
    print(min2[0] - max1[0])
    print(min1[1] - max2[1])
    print(min2[1] - max1[1])
    
    # 0 axis; boundary with 2 < 1
    if (min1[0] - max2[0] >= 0 and min1[0] - max2[0] < 0.5):
        print("joining along 0 boundary")
        print("first greater than second")
        bound1 = Z1[:,0] < min1[0] + margin*sigma
        bound2 = Z2[:,0] > max2[0] - margin*sigma
        Zopt = np.vstack((Z1[bound1,:],Z2[bound2,:]))
        nuZopt = np.vstack((nuZ1[bound1,:],nuZ2[bound2,:]))
        Z = np.vstack((Z1[~bound1,:],Z2[~bound2,:]))
        nuZ = np.vstack((nuZ1[~bound1,:],nuZ2[~bound2,:]))
    
    # 0 axis; boundary with 1 < 2
    elif (min2[0] - max1[0] >= 0 and min2[0] - max1[0] < 0.5):
        print("joining along 0 boundary")
        print("second greater than first")
        bound1 = Z1[:,0] > max1[0] - margin*sigma
        bound2 = Z2[:,0] < min2[0] + margin*sigma
        Zopt = np.vstack((Z1[bound1,:],Z2[bound2,:]))
        nuZopt = np.vstack((nuZ1[bound1,:],nuZ2[bound2,:]))
        Z = np.vstack((Z1[~bound1,:],Z2[~bound2,:]))
        nuZ = np.vstack((nuZ1[~bound1,:],nuZ2[~bound2,:]))
        
    # 1 axis; boundary with 2 < 1    
    elif (min1[1] - max2[1] >= 0 and min1[1] - max2[1] < 0.5):
        print("joining along 1 boundary")
        print("first greater than second")
        bound1 = Z1[:,1] < min1[1] + margin*sigma
        bound2 = Z2[:,1] > max2[1] - margin*sigma
        Zopt = np.vstack((Z1[bound1,:],Z2[bound2,:]))
        nuZopt = np.vstack((nuZ1[bound1,:],nuZ2[bound2,:]))
        Z = np.vstack((Z1[~bound1,:],Z2[~bound2,:]))
        nuZ = np.vstack((nuZ1[~bound1,:],nuZ2[~bound2,:]))
        
    # 1 axis; boundary with 1 < 2
    elif (min2[1] - max1[1] >= 0 and min2[1] - max1[1] < 0.5):
        print("joining along 1 boundary")
        print("second greater than first")
        bound1 = Z1[:,1] > max1[1] - margin*sigma
        bound2 = Z2[:,1] < min2[1] + margin*sigma
        Zopt = np.vstack((Z1[bound1,:],Z2[bound2,:]))
        nuZopt = np.vstack((nuZ1[bound1,:],nuZ2[bound2,:]))
        Z = np.vstack((Z1[~bound1,:],Z2[~bound2,:]))
        nuZ = np.vstack((nuZ1[~bound1,:],nuZ2[~bound2,:]))
    
    else:
        return
    
    X = np.vstack((X1,X2))
    nuX = np.vstack((x1Info['nu_X'],x2Info['nu_X']))
    
    # Check that you have divided Z1 + Z2 into Zopt and Z 
    print("total Z's before optimizing are " + str(Z1.shape[0] + Z2.shape[0]))
    print("total Z's after split and combine are " + str(Zopt.shape[0] + Z.shape[0]))
    
    onu_X, onuZopt, onuZ, indsToKeep = oneHot(nuX,nuZopt,nuZ)
    
    def make_ranges(Xlist, Zo, Zn, epsX, epsZo, epsZn, sig):
      # Here X and Z are torch tensors
        # Base the keep criteria on the non-optimizing eps (e.g. X or Zn)
        a = np.sqrt(3)
        rangesZoXList = []
        X_labelsList = []
        Zo_labels = grid_cluster(Zo, epsZo) 
        Zo_ranges, Zo_centroids, _ = cluster_ranges_centroids(Zo, Zo_labels)
        D = ((Zo_centroids[:, None, :] - Zo_centroids[None, :, :]) ** 2).sum(dim=2)
        keep = D <(a*epsZo+4* sig) ** 2
        rangesZoZo_ij = from_matrix(Zo_ranges, Zo_ranges, keep)
        areas = (Zo_ranges[:, 1] - Zo_ranges[:, 0])[:, None] * (Zo_ranges[:, 1] 
                        - Zo_ranges[:, 0])[None, :]
        total_area = areas.sum()  # should be equal to N*M
        sparse_area = areas[keep].sum()
        print(
        "We keep {:.2e}/{:.2e} = {:2d}% of the original kernel matrix.".format(
          sparse_area, total_area, int(100 * sparse_area / total_area)))
        print("")
        
        Zn_labels = grid_cluster(Zn,epsZn)
        Zn_ranges, Zn_centroids, _ = cluster_ranges_centroids(Zn,Zn_labels)
        D = ((Zo_centroids[:,None,:] - Zn_centroids[None,:,:])**2).sum(dim=2)
        keep = D < (a*(epsZn/2.0 + epsZo/2.0)+4*sig)**2
        rangesZoZn_ij = from_matrix(Zo_ranges,Zn_ranges,keep)
        areas = (Zo_ranges[:,1] - Zo_ranges[:,0])[:,None]*(Zn_ranges[:,1] - Zn_ranges[:,0])[None,:]
        total_area = areas.sum()
        sparse_area = areas[keep].sum()
        print("We keep {:.2e}/{:.2e} = {:2d}% of the original kernel matrix.".format(
              sparse_area, total_area, int(100 * sparse_area / total_area)))
        print("")

        for cc in range(len(Xlist)):
            X = Xlist[cc]
            if len(X) < 1:
                rangesZoXList.append([])
                X_labelsList.append([])
            eps = epsX[cc]
            X_labels = grid_cluster(X, eps)
            X_labelsList.append(X_labels)
            X_ranges, X_centroids, _ = cluster_ranges_centroids(X, X_labels)

            D = ((Zo_centroids[:, None, :] - X_centroids[None, :, :]) ** 2).sum(dim=2)
            keep = D <(a*(eps/2.0 + epsZo/2.0)+4* sig) ** 2
            rangesZoX_ij = from_matrix(Zo_ranges, X_ranges, keep)
            areas = (Zo_ranges[:, 1] - Zo_ranges[:, 0])[:, None] * (X_ranges[:, 1] 
                            - X_ranges[:, 0])[None, :]
            total_area = areas.sum()  # should be equal to N*M
            sparse_area = areas[keep].sum()
            print(
            "We keep {:.2e}/{:.2e} = {:2d}% of the original kernel matrix.".format(
              sparse_area, total_area, int(100 * sparse_area / total_area)))
            print("")
            #print("memory (bytes)")
            #print(process.memory_info().rss)  # in bytes
            rangesZoXList.append(rangesZoX_ij)
            
        return rangesZoXList, rangesZoZn_ij, rangesZoZo_ij, X_labelsList, Zo_labels, Zn_labels

    def make_loss(tXlist, tZn, tnuZn, len_Z, dim_nu_Z, rangesZoX, rangesZoZo_ij, rangesZoZn_ij):

        def loss(tZal_Z):
            # z operation is still assumed to be the same (mixed distribution of labels)
            LZ_i, LZ_j = Vi(tZal_Z[0:3*len_Z].view(-1,3)), Vj(tZal_Z[0:3*len_Z].view(-1,3))
    
            Lnu_Z_i= Vi(tZal_Z[3*len_Z::].view(-1,dim_nu_Z)**2)
            Lnu_Z_j= Vj(tZal_Z[3*len_Z::].view(-1,dim_nu_Z)**2)

            DZZ_ij = ((LZ_i - LZ_j)**2/sig**2).sum(dim=2)  
            KZZ_ij = (- DZZ_ij).exp()  
            PZZ_ij = (Lnu_Z_i*Lnu_Z_j).sum(dim=2)
            KPZZ_ij = KZZ_ij*PZZ_ij
            KPZZ_ij.ranges = rangesZoZo_ij
            
            LZn_j = LazyTensor(tZn[None,:,:])
            LnuZn_j = LazyTensor(tnuZn[None,:,:])
            DZZn_ij = ((LZ_i - LZn_j)**2/sig**2).sum(dim=2)
            KZZn_ij = (- DZZn_ij).exp()
            PZZn_ij = (Lnu_Z_i*LnuZn_j).sum(dim=2)
            KPZZn_ij = KZZn_ij*PZZn_ij
            KPZZn_ij.ranges = rangesZoZn_ij
            
            L = KPZZ_ij.sum(dim=1).sum() + 2*KPZZn_ij.sum(dim=1).sum()
            print("shape of L is ")
            print(L.shape)
            #torch.cuda.empty_cache()
            print("memory (bytes)")
            print(process.memory_info().rss)  # in bytes
            GPUtil.showUtilization()
            L.backward()
            print("memory (bytes)")
            print(process.memory_info().rss)  # in bytes
            GPUtil.showUtilization()
            for lab in range(len(tXlist)):
                rangesZX_ij = rangesZoX[lab] # Getting the ranges
                if len(rangesZX_ij) < 1:
                    continue
                tX = tXlist[lab]
                #LX_i = LazyTensor(tX[:,None,:])
                LX_j = LazyTensor(tX[None,:,:])
                DZX_ij = ((LZ_i - LX_j)**2/sig**2).sum(dim=2) 
                KZX_ij = (- DZX_ij).exp() 
            #PZX_ij = (Lnu_Z_i*Lnu_X_j).sum(dim=2) # 
                KPZX_ij = Vi(tZal_Z[3*len_Z::].view(-1,dim_nu_Z)[...,lab][...,None]**2)*KZX_ij
                # KPZX_ij = Lnu_Z_i.elem(lab)*KZX_ij # Alternative 
                KPZX_ij.ranges = rangesZX_ij
                Es = -2*KPZX_ij.sum(dim=1).sum()
                Es.backward()
                L += Es
            return L.detach()
        return loss 
    
    def make_loss2(tXlist,tZ, tZn, tnuZn, dim_nu_Z, rangesZoX, rangesZoZo_ij, rangesZoZn_ij):

        LZ_i, LZ_j= Vi(tZ), Vj(tZ)
        DZZ_ij = ((LZ_i - LZ_j)**2/sig**2).sum(dim=2)  
        KZZ_ij = (- DZZ_ij).exp()
        
        LZn_j = LazyTensor(tZn[None,:,:])
        LnuZn_j = LazyTensor(tnuZn[None,:,:])
        DZZn_ij = ((LZ_i - LZn_j)**2/sig**2).sum(dim=2)
        KZZn_ij = (- DZZn_ij).exp()


        def loss(tal_Z): 
            Lnu_Z_i, Lnu_Z_j = Vi(tal_Z.view(-1,dim_nu_Z)**2), Vj(tal_Z.view(-1,dim_nu_Z)**2)

            PZZ_ij = (Lnu_Z_i*Lnu_Z_j).sum(dim=2)
            KPZZ_ij = KZZ_ij*PZZ_ij
            KPZZ_ij.ranges = rangesZoZo_ij
            
            PZZn_ij = (Lnu_Z_i*LnuZn_j).sum(dim=2)
            KPZZn_ij = KZZn_ij*PZZn_ij
            KPZZn_ij.ranges = rangesZoZn_ij
            
            L = KPZZ_ij.sum(dim=1).sum() + 2*KPZZn_ij.sum(dim=1).sum()

            print("shape of L is ")
            print(L.shape)
            #torch.cuda.empty_cache()
            print("memory (bytes)")
            print(process.memory_info().rss)  # in bytes
            GPUtil.showUtilization()
            L.backward()
            print("memory (bytes)")
            print(process.memory_info().rss)  # in bytes
            GPUtil.showUtilization()
            for lab in range(len(tXlist)):
                rangesZX_ij = rangesZoX[lab] # Getting the ranges
                if len(rangesZX_ij) < 1:
                    continue
                tX = tXlist[lab]
                #LX_i = LazyTensor(tX[:,None,:])
                LX_j = LazyTensor(tX[None,:,:])
                DZX_ij = ((LZ_i - LX_j)**2/sig**2).sum(dim=2) 
                KZX_ij = (- DZX_ij).exp() 
            #PZX_ij = (Lnu_Z_i*Lnu_X_j).sum(dim=2) # 
                KPZX_ij = Vi(tal_Z.view(-1,dim_nu_Z)[...,lab][...,None]**2)*KZX_ij
                #KPZX_ij = Lnu_Z_i.elem(lab)*KZX_ij # alternative 
                KPZX_ij.ranges = rangesZX_ij
                Es = -2*KPZX_ij.sum(dim=1).sum()
                L += Es
                Es.backward() # deleting retain graph                 
                #print("passed Es loss for " + str(lab) + " out of " + str(len(tXlist)-1))
                #Lback.append(E.sum().backward())
            return L.detach()
        print("memory (bytes)")
        print(process.memory_info().rss)  # in bytes
        return loss

    Xlist = groupXbyLabel(onu_X,X)

    # make Epislon List
    def makeEps(Xlist,Zo,Z,Nmax,Npart):
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
        
        denZo = min(Zo.shape[0]/Npart,Nmax)
        volZo = np.prod(np.max(Zo,axis=0) - np.min(Zo,axis=0) + 0.001)
        epsZo = np.cbrt(volZo/denZo)
        print("Zo\t" + str(volZ) + "\t" + str(Zo.shape[0]) + "\t" + str(denZo) + "\t" + str(epsZo))

        for X in Xlist:
            if X.shape[0] < 1:
                epsList.append(-1)
                continue
            den = min(X.shape[0]/Npart,Nmax)
            volX = np.prod(np.max(X,axis=0)-np.min(X,axis=0) + 0.001)
            epsX = np.cbrt(volX/den)
            epsList.append(epsX)
            print("X\t" + str(volX) + "\t" + str(X.shape[0]) + "\t" + str(den) + "\t" + str(epsX))
        return epsList,epsZo,epsZ
    
    temptime = time.time()
    epsX,epsZo,epsZ = makeEps(Xlist,Zopt,Z,Nmax,Npart)
    print("time for making epsilon is " + str(time.time()-temptime)) 
    
    # make tensors for each
    tXlist = []
    for Xl in Xlist:
        tXlist.append(torch.tensor(Xl).type(dtype))
        
    tZo = torch.tensor(Zopt).type(dtype)
    tnu_Zo = torch.tensor(onuZopt).type(dtype)
    tZ = torch.tensor(Z).type(dtype)
    tnu_Z = torch.tensor(onuZ).type(dtype)

    # Computes ranges and labels for the grid
    print("Making ranges")
    temptime = time.time()
    rangesZoXList, rangesZoZn_ij, rangesZoZo_ij, X_labelsList, Zo_labels, Zn_labels = make_ranges(tXlist, tZo, tZ, epsX, epsZo,epsZ,sig)
    print("time for making ranges is " + str(time.time() - temptime))
    
    # Sorts X and nu_X
    print("Sorting X and nu_X")
    temptime = time.time()
    for k in range(len(tXlist)):
        if len(X_labelsList[k]) < 1:
            continue
        tXlist[k], _ = sort_clusters(tXlist[k], X_labelsList[k]) # sorting the labels
    print("time for sorting X is " + str(time.time() - temptime))

    #  Sorts Z and nu_Z
    print("Sorting Z and nu_Z")
    temptime = time.time()
    tZ, _ = sort_clusters(tZ, Zn_labels) # sorting the labels
    tnu_Z, _ = sort_clusters(tnu_Z, Zn_labels)
    print("time for sorting Z is " + str(time.time() - temptime))
    len_Z, dim_nu_Z = onuZ.shape
    
    tZo,_ = sort_clusters(tZo, Zo_labels)
    tnu_Zo, _ = sort_clusters(tnu_Zo,Zo_labels)
    len_Zo, dim_nu_Zo = onuZopt.shape

    # Optimization

    def optimize(tZo, tnu_Zo, nb_iter = 20, flag = 'all'):
        if flag == 'all':
            temptime = time.time()
            #loss = make_loss(tXlist, len_Z, dim_nu_Z, rangesXXList, rangesZZ_ij, rangesZXList)
            loss = make_loss(tXlist, tZ, tnu_Z, len_Zo, dim_nu_Zo, rangesZoXList, rangesZoZo_ij, rangesZoZn_ij)
            print("time for making loss is " + str(time.time() - temptime))
            p0 = torch.cat((tZo.flatten(),tnu_Zo.pow(0.5).flatten()),0).requires_grad_(True)
        else:
            temptime = time.time()
            #loss = make_loss2(tXlist, tZ, dim_nu_Z, rangesXXList, rangesZZ_ij, rangesZXList)
            loss = make_loss2(tXlist,tZo, tZ, tnu_Z, dim_nu_Zo, rangesZoXList, rangesZoZo_ij, rangesZoZn_ij)
            print("time for making loss 2 is " + str(time.time() - temptime))
            p0 = tnu_Zo.pow(0.5).flatten().clone().requires_grad_(True)

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
            L = loss(p0)
            print("error is ", L.detach().cpu().numpy())
            #print("relative error loss", L.detach().cpu().numpy()/c.detach().cpu().numpy())
            lossTrack.append(L.detach().cpu().numpy())
            #tot.backward()
            return L

        for i in range(nb_iter):
            print("it ", i, ": ", end="")
            print(torch.cuda.memory_allocated(0))
            GPUtil.showUtilization()
            temptime = time.time()
            optimizer.step(closure)
            print("time to take a step is " + str(time.time() - temptime))
            #torch.cuda.empty_cache()

        if flag == 'all':
            tnZo = p0[0:3*len_Zo].detach().view(-1,3)
            tnnu_Zo = p0[3*len_Zo::].detach().view(-1,dim_nu_Zo)**2
        else:
            tnZo = tZo
            tnnu_Zo = p0.detach().view(-1,dim_nu_Zo)**2
        return tnZo, tnnu_Zo

    print("Starting Optim")
    print("sum tnu_Z before", tnu_Zo.sum())
    tZo, tnu_Zo = optimize(tZo, tnu_Zo, nb_iter = nb_iter0, flag = '')
    tnZo, tnnu_Zo = tZo, tnu_Zo
    #torch.cuda.empty_cache()
    tnZo, tnnu_Zo = optimize(tZo, tnu_Zo, nb_iter = nb_iter1, flag = 'all')
    print("sum tnnu_Z after", tnnu_Zo.sum())

    nZ = tnZo.detach().cpu().numpy()
    nnu_Z = tnnu_Zo.detach().cpu().numpy()
    
    # determine amount by which particles have moved and by which distributions have changed (squared error distance)
    distMove = np.sqrt(np.sum((Zopt - nZ)**2,axis=-1))
    distAlt = np.sqrt(np.sum((onuZopt-nnu_Z)**2,axis=-1))
    fig,ax = plt.subplots(2,1)
    ax[0].hist(distMove)
    ax[1].hist(distAlt)
    ax[0].set_title('Distance Moved')
    ax[1].set_title('Distribution Changed')
    fig.savefig(outpath+'_optimalZnu_ZwC' + str(C) + '_sig' + str(sig) + '_Nmax' + str(Nmax) + '_Npart' + str(Npart) + '_distancesStitch.png',dpi=300)
    np.savez(outpath+'_optimalZnu_ZwC' + str(C) + '_sig' + str(sig) + '_Nmax' + str(Nmax) + '_Npart' + str(Npart) + '_distancesStitch.npz',Z=Zopt,nZ=nZ,nu_Z=onuZopt,nnu_Z=nnu_Z,distMove=distMove,distAlt=distAlt)
    
    nnu_Z = reverseOneHot(nnu_Z,indsToKeep,maxV)
    print("nnu_Z shape should be number of particles by maxV")
    print(nnu_Z.shape)
    
    nZcomb = np.vstack((nZ,Z))
    print("should be equal ")
    print(str(nZcomb.shape[0]) + " vs . " + str(Z1.shape[0] + Z2.shape[0]))
    nuZcomb = np.vstack((nnu_Z,nuZ))
    print("nu Z shape is " + str(nuZcomb.shape))
    nZedit = np.vstack((np.ones((nZ.shape[0],1)),np.zeros((Z.shape[0],1))))
    
    maxInd = np.argmax(nuZcomb,axis=-1)+1
    vtf.writeVTK(nZcomb,[maxInd,np.sum(nuZcomb,axis=-1),nZedit],['MAX_VAL_NU','TOTAL_MASS','OPTIMIZED'],outpath+'_optimalZnu_ZAllwC' + str(C) + '_sig' + str(sig) + '_Nmax' + str(Nmax) + '_Npart' + str(Npart) +'.vtk',polyData=None)
    fZ = outpath+'_optimalZnu_ZAllwC' + str(C) + '_sig' + str(sig) + '_Nmax' + str(Nmax) + '_Npart' + str(Npart) +'.npz'
    np.savez(fZ,Z=nZcomb, nu_Z=nuZcomb,Zo=nZ,nu_Zo=nnu_Z)

    # Not good !!!! -- removes low mass particles 
    mean = np.mean(nnu_Z.sum(axis=1))
    c = nnu_Z.sum(axis=1)>0.1*mean
    print("number of orig particles " + str(nnu_Z.shape[0]))
    print("number of new particles after remove low mass " + str(np.sum(c)))
    nZ = nZ[c,:]
    nnu_Z = nnu_Z[c,:]
    
    nZcomb = np.vstack((nZ,Z))
    nuZcomb = np.vstack((nnu_Z,nuZ))
    nZedit = np.vstack((np.ones((nZ.shape[0],1)),np.zeros((Z.shape[0],1))))

    # Display of the result (in reduced integer set)
    maxInd = np.argmax(nuZcomb,axis=-1)+1
    vtf.writeVTK(nZcomb,[maxInd,np.sum(nuZcomb,axis=-1),nZedit],['MAX_VAL_NU','TOTAL_MASS','OPTIMIZED'],outpath+'_optimalZnu_ZwC' + str(C) + '_sig' + str(sig) + '_Nmax' + str(Nmax) + '_Npart' + str(Npart) +'.vtk',polyData=None)
    np.savez(outpath+'_optimalZnu_ZwC' + str(C) + '_sig' + str(sig) + '_Nmax' + str(Nmax) + '_Npart' + str(Npart) +'.npz',Z=nZcomb, nu_Z=nuZcomb,Zo=nZ,nu_Zo=nnu_Z)
    
    fX = outpath+'_XnuX.npz'
    np.savez(fX,X=X, nu_X=nuX)

    # remove to see if release memory 
    f,ax = plt.subplots()
    ax.plot(np.arange(len(lossTrack)),np.asarray(lossTrack))
    ax.set_xlabel('iterations')
    ax.set_ylabel('cost')
    f.savefig(outpath+ '_optimalZcost_sig' + str(sig) + '_C' + str(C) + '_Nmax' + str(Nmax) + '_Npart' + str(Npart) + '.png',dpi=300)
    
    #return X, nuX, nZcomb, nuZcomb, fX, fZ
    return fX, fZ
    
def stitchSlabs(x1,x2,z1,z2,outpath,sigma,nb_iter0,nb_iter1,margin=2.0,Nmax=2000.0,Npart=50000.0,maxV=673,optMethod='LBFGS',ax=2):
    '''
    Takes in 2 slabs and smoothes the boundary between them
    
    Assumes the slabs are oriented along last axis
    '''
    sig = sigma
    process = psutil.Process(os.getpid())
    lossTrack = []
    C=1.2
    
    print("working on X files")
    print(x1)
    print(x2)
    
    print("working on Z files")
    print(z1)
    print(z2)
    
    # First determine where boundaries are and whether there is a boundary to optimize 
    x1Info = np.load(x1)
    X1 = x1Info['X']
    x2Info = np.load(x2)
    X2 = x2Info['X']
    min1 = np.min(X1,axis=0)
    max1 = np.max(X1,axis=0)
    min2 = np.min(X2,axis=0)
    max2 = np.max(X2,axis=0)
    
    z1Info = np.load(z1)
    Z1 = z1Info['Z']
    nuZ1 = z1Info['nu_Z']
    
    z2Info = np.load(z2)
    Z2 = z2Info['Z']
    nuZ2 = z2Info['nu_Z']
    
    # 2 < 1
    if (min1[ax] - max2[ax] >= 0):
        print("first greater than second")
        bound1 = Z1[:,ax] < min1[ax] + margin*sigma
        bound2 = Z2[:,ax] > max2[ax] - margin*sigma
        Zopt = np.vstack((Z1[bound1,:],Z2[bound2,:]))
        nuZopt = np.vstack((nuZ1[bound1,:],nuZ2[bound2,:]))
        Z = np.vstack((Z1[~bound1,:],Z2[~bound2,:]))
        nuZ = np.vstack((nuZ1[~bound1,:],nuZ2[~bound2,:]))
        lbOpt = max2[ax] - margin*sigma - 2.0*margin*sigma
        ubOpt = min1[ax] + margin*sigma + 2.0*margin*sigma
    # 1 < 2
    else:
        print("second greater than first")
        bound1 = Z1[:,ax] > max1[ax] - margin*sigma
        bound2 = Z2[:,ax] < min2[ax] + margin*sigma
        Zopt = np.vstack((Z1[bound1,:],Z2[bound2,:]))
        nuZopt = np.vstack((nuZ1[bound1,:],nuZ2[bound2,:]))
        Z = np.vstack((Z1[~bound1,:],Z2[~bound2,:]))
        nuZ = np.vstack((nuZ1[~bound1,:],nuZ2[~bound2,:]))
        lbOpt = max1[ax] - margin*sigma - 2.0*margin*sigma
        ubOpt = min2[ax] + margin*sigma + 2.0*margin*sigma

    X = np.vstack((X1,X2))
    nuX = np.vstack((x1Info['nu_X'],x2Info['nu_X']))
    xKeep = (X[:,ax] > ubOpt) + (X[:,ax] < lbOpt)
    zKeep = (Z[:,ax] > ubOpt) + (Z[:,ax] < lbOpt)
    
    Xkeep = X[xKeep,:]
    nuXkeep = nuX[xKeep,:]
    X = X[~xKeep,:]
    nuX = nuX[~xKeep,:]
    Zkeep = Z[zKeep,:]
    nuZkeep = nuZ[zKeep,:]
    Z = Z[~zKeep,:]
    nuZ = nuZ[~zKeep,:]
    
    print("shapes of keep vs not in X")
    print(Xkeep.shape)
    print(X.shape)
    
    
    # Check that you have divided Z1 + Z2 into Zopt and Z 
    print("total Z's before optimizing are " + str(Z1.shape[0] + Z2.shape[0]))
    print("total Z's after split and combine are " + str(Zopt.shape[0] + Z.shape[0]))
    
    onu_X, onuZopt, onuZ, indsToKeep = oneHot(nuX,nuZopt,nuZ)
    
    def make_ranges(Xlist, Zo, Zn, epsX, epsZo, epsZn, sig):
      # Here X and Z are torch tensors
        # Base the keep criteria on the non-optimizing eps (e.g. X or Zn)
        a = np.sqrt(3)
        rangesZoXList = []
        X_labelsList = []
        Zo_labels = grid_cluster(Zo, epsZo) 
        Zo_ranges, Zo_centroids, _ = cluster_ranges_centroids(Zo, Zo_labels)
        D = ((Zo_centroids[:, None, :] - Zo_centroids[None, :, :]) ** 2).sum(dim=2)
        keep = D <(a*epsZo+4* sig) ** 2
        rangesZoZo_ij = from_matrix(Zo_ranges, Zo_ranges, keep)
        areas = (Zo_ranges[:, 1] - Zo_ranges[:, 0])[:, None] * (Zo_ranges[:, 1] 
                        - Zo_ranges[:, 0])[None, :]
        total_area = areas.sum()  # should be equal to N*M
        sparse_area = areas[keep].sum()
        print(
        "We keep {:.2e}/{:.2e} = {:2d}% of the original kernel matrix.".format(
          sparse_area, total_area, int(100 * sparse_area / total_area)))
        print("")
        
        Zn_labels = grid_cluster(Zn,epsZn)
        Zn_ranges, Zn_centroids, _ = cluster_ranges_centroids(Zn,Zn_labels)
        D = ((Zo_centroids[:,None,:] - Zn_centroids[None,:,:])**2).sum(dim=2)
        keep = D < (a*(epsZn/2.0 + epsZo/2.0)+4*sig)**2
        rangesZoZn_ij = from_matrix(Zo_ranges,Zn_ranges,keep)
        areas = (Zo_ranges[:,1] - Zo_ranges[:,0])[:,None]*(Zn_ranges[:,1] - Zn_ranges[:,0])[None,:]
        total_area = areas.sum()
        sparse_area = areas[keep].sum()
        print("We keep {:.2e}/{:.2e} = {:2d}% of the original kernel matrix.".format(
              sparse_area, total_area, int(100 * sparse_area / total_area)))
        print("")

        for cc in range(len(Xlist)):
            X = Xlist[cc]
            if len(X) < 1:
                rangesZoXList.append([])
                X_labelsList.append([])
                continue
            eps = epsX[cc]
            X_labels = grid_cluster(X, eps)
            X_labelsList.append(X_labels)
            X_ranges, X_centroids, _ = cluster_ranges_centroids(X, X_labels)

            D = ((Zo_centroids[:, None, :] - X_centroids[None, :, :]) ** 2).sum(dim=2)
            keep = D <(a*(eps/2.0 + epsZo/2.0)+4* sig) ** 2
            rangesZoX_ij = from_matrix(Zo_ranges, X_ranges, keep)
            areas = (Zo_ranges[:, 1] - Zo_ranges[:, 0])[:, None] * (X_ranges[:, 1] 
                            - X_ranges[:, 0])[None, :]
            total_area = areas.sum()  # should be equal to N*M
            sparse_area = areas[keep].sum()
            print(
            "We keep {:.2e}/{:.2e} = {:2d}% of the original kernel matrix.".format(
              sparse_area, total_area, int(100 * sparse_area / total_area)))
            print("")
            #print("memory (bytes)")
            #print(process.memory_info().rss)  # in bytes
            rangesZoXList.append(rangesZoX_ij)
            
        return rangesZoXList, rangesZoZn_ij, rangesZoZo_ij, X_labelsList, Zo_labels, Zn_labels

    def make_loss(tXlist, tZn, tnuZn, len_Z, dim_nu_Z, rangesZoX, rangesZoZo_ij, rangesZoZn_ij):

        def loss(tZal_Z):
            # z operation is still assumed to be the same (mixed distribution of labels)
            LZ_i, LZ_j = Vi(tZal_Z[0:3*len_Z].view(-1,3)), Vj(tZal_Z[0:3*len_Z].view(-1,3))
    
            Lnu_Z_i= Vi(tZal_Z[3*len_Z::].view(-1,dim_nu_Z)**2)
            Lnu_Z_j= Vj(tZal_Z[3*len_Z::].view(-1,dim_nu_Z)**2)

            DZZ_ij = ((LZ_i - LZ_j)**2/sig**2).sum(dim=2)  
            KZZ_ij = (- DZZ_ij).exp()  
            PZZ_ij = (Lnu_Z_i*Lnu_Z_j).sum(dim=2)
            KPZZ_ij = KZZ_ij*PZZ_ij
            KPZZ_ij.ranges = rangesZoZo_ij
            
            LZn_j = LazyTensor(tZn[None,:,:])
            LnuZn_j = LazyTensor(tnuZn[None,:,:])
            DZZn_ij = ((LZ_i - LZn_j)**2/sig**2).sum(dim=2)
            KZZn_ij = (- DZZn_ij).exp()
            PZZn_ij = (Lnu_Z_i*LnuZn_j).sum(dim=2)
            KPZZn_ij = KZZn_ij*PZZn_ij
            KPZZn_ij.ranges = rangesZoZn_ij
            
            L = KPZZ_ij.sum(dim=1).sum() + 2*KPZZn_ij.sum(dim=1).sum()
            print("shape of L is ")
            print(L.shape)
            #torch.cuda.empty_cache()
            print("memory (bytes)")
            print(process.memory_info().rss)  # in bytes
            GPUtil.showUtilization()
            L.backward()
            print("memory (bytes)")
            print(process.memory_info().rss)  # in bytes
            GPUtil.showUtilization()
            for lab in range(len(tXlist)):
                rangesZX_ij = rangesZoX[lab] # Getting the ranges
                if len(rangesZX_ij) < 1:
                    continue
                tX = tXlist[lab]
                #LX_i = LazyTensor(tX[:,None,:])
                LX_j = LazyTensor(tX[None,:,:])
                DZX_ij = ((LZ_i - LX_j)**2/sig**2).sum(dim=2) 
                KZX_ij = (- DZX_ij).exp() 
            #PZX_ij = (Lnu_Z_i*Lnu_X_j).sum(dim=2) # 
                KPZX_ij = Vi(tZal_Z[3*len_Z::].view(-1,dim_nu_Z)[...,lab][...,None]**2)*KZX_ij
                # KPZX_ij = Lnu_Z_i.elem(lab)*KZX_ij # Alternative 
                KPZX_ij.ranges = rangesZX_ij
                Es = -2*KPZX_ij.sum(dim=1).sum()
                Es.backward()
                L += Es
            return L.detach()
        return loss 
    
    def make_loss2(tXlist,tZ, tZn, tnuZn, dim_nu_Z, rangesZoX, rangesZoZo_ij, rangesZoZn_ij):

        LZ_i, LZ_j= Vi(tZ), Vj(tZ)
        DZZ_ij = ((LZ_i - LZ_j)**2/sig**2).sum(dim=2)  
        KZZ_ij = (- DZZ_ij).exp()
        
        LZn_j = LazyTensor(tZn[None,:,:])
        LnuZn_j = LazyTensor(tnuZn[None,:,:])
        DZZn_ij = ((LZ_i - LZn_j)**2/sig**2).sum(dim=2)
        KZZn_ij = (- DZZn_ij).exp()


        def loss(tal_Z): 
            Lnu_Z_i, Lnu_Z_j = Vi(tal_Z.view(-1,dim_nu_Z)**2), Vj(tal_Z.view(-1,dim_nu_Z)**2)

            PZZ_ij = (Lnu_Z_i*Lnu_Z_j).sum(dim=2)
            KPZZ_ij = KZZ_ij*PZZ_ij
            KPZZ_ij.ranges = rangesZoZo_ij
            
            PZZn_ij = (Lnu_Z_i*LnuZn_j).sum(dim=2)
            KPZZn_ij = KZZn_ij*PZZn_ij
            KPZZn_ij.ranges = rangesZoZn_ij
            
            L = KPZZ_ij.sum(dim=1).sum() + 2*KPZZn_ij.sum(dim=1).sum()

            print("shape of L is ")
            print(L.shape)
            #torch.cuda.empty_cache()
            print("memory (bytes)")
            print(process.memory_info().rss)  # in bytes
            GPUtil.showUtilization()
            L.backward()
            print("memory (bytes)")
            print(process.memory_info().rss)  # in bytes
            GPUtil.showUtilization()
            for lab in range(len(tXlist)):
                rangesZX_ij = rangesZoX[lab] # Getting the ranges
                if len(rangesZX_ij) < 1:
                    continue
                tX = tXlist[lab]
                #LX_i = LazyTensor(tX[:,None,:])
                LX_j = LazyTensor(tX[None,:,:])
                DZX_ij = ((LZ_i - LX_j)**2/sig**2).sum(dim=2) 
                KZX_ij = (- DZX_ij).exp() 
            #PZX_ij = (Lnu_Z_i*Lnu_X_j).sum(dim=2) # 
                KPZX_ij = Vi(tal_Z.view(-1,dim_nu_Z)[...,lab][...,None]**2)*KZX_ij
                #KPZX_ij = Lnu_Z_i.elem(lab)*KZX_ij # alternative 
                KPZX_ij.ranges = rangesZX_ij
                Es = -2*KPZX_ij.sum(dim=1).sum()
                L += Es
                Es.backward() # deleting retain graph                 
                #print("passed Es loss for " + str(lab) + " out of " + str(len(tXlist)-1))
                #Lback.append(E.sum().backward())
            return L.detach()
        print("memory (bytes)")
        print(process.memory_info().rss)  # in bytes
        return loss

    Xlist = groupXbyLabel(onu_X,X)

    # make Epislon List
    def makeEps(Xlist,Zo,Z,Nmax,Npart):
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
        
        denZo = min(Zo.shape[0]/Npart,Nmax)
        volZo = np.prod(np.max(Zo,axis=0) - np.min(Zo,axis=0) + 0.001)
        epsZo = np.cbrt(volZo/denZo)
        print("Zo\t" + str(volZ) + "\t" + str(Zo.shape[0]) + "\t" + str(denZo) + "\t" + str(epsZo))

        for X in Xlist:
            den = min(X.shape[0]/Npart,Nmax)
            if X.shape[0] < 1:
                epsList.append(-1)
                continue
            volX = np.prod(np.max(X,axis=0)-np.min(X,axis=0) + 0.001)
            epsX = np.cbrt(volX/den)
            epsList.append(epsX)
            print("X\t" + str(volX) + "\t" + str(X.shape[0]) + "\t" + str(den) + "\t" + str(epsX))
        return epsList,epsZo,epsZ
    
    temptime = time.time()
    epsX,epsZo,epsZ = makeEps(Xlist,Zopt,Z,Nmax,Npart)
    print("time for making epsilon is " + str(time.time()-temptime)) 
    
    # make tensors for each
    tXlist = []
    for Xl in Xlist:
        tXlist.append(torch.tensor(Xl).type(dtype))
        
    tZo = torch.tensor(Zopt).type(dtype)
    tnu_Zo = torch.tensor(onuZopt).type(dtype)
    tZ = torch.tensor(Z).type(dtype)
    tnu_Z = torch.tensor(onuZ).type(dtype)

    # Computes ranges and labels for the grid
    print("Making ranges")
    temptime = time.time()
    rangesZoXList, rangesZoZn_ij, rangesZoZo_ij, X_labelsList, Zo_labels, Zn_labels = make_ranges(tXlist, tZo, tZ, epsX, epsZo,epsZ,sig)
    print("time for making ranges is " + str(time.time() - temptime))
    
    # Sorts X and nu_X
    print("Sorting X and nu_X")
    temptime = time.time()
    for k in range(len(tXlist)):
        if len(X_labelsList[k]) < 1:
            continue
        tXlist[k], _ = sort_clusters(tXlist[k], X_labelsList[k]) # sorting the labels
    print("time for sorting X is " + str(time.time() - temptime))

    #  Sorts Z and nu_Z
    print("Sorting Z and nu_Z")
    temptime = time.time()
    tZ, _ = sort_clusters(tZ, Zn_labels) # sorting the labels
    tnu_Z, _ = sort_clusters(tnu_Z, Zn_labels)
    print("time for sorting Z is " + str(time.time() - temptime))
    len_Z, dim_nu_Z = onuZ.shape
    
    tZo,_ = sort_clusters(tZo, Zo_labels)
    tnu_Zo, _ = sort_clusters(tnu_Zo,Zo_labels)
    len_Zo, dim_nu_Zo = onuZopt.shape

    # Optimization

    def optimize(tZo, tnu_Zo, nb_iter = 20, flag = 'all'):
        if flag == 'all':
            temptime = time.time()
            #loss = make_loss(tXlist, len_Z, dim_nu_Z, rangesXXList, rangesZZ_ij, rangesZXList)
            loss = make_loss(tXlist, tZ, tnu_Z, len_Zo, dim_nu_Zo, rangesZoXList, rangesZoZo_ij, rangesZoZn_ij)
            print("time for making loss is " + str(time.time() - temptime))
            p0 = torch.cat((tZo.flatten(),tnu_Zo.pow(0.5).flatten()),0).requires_grad_(True)
        else:
            temptime = time.time()
            #loss = make_loss2(tXlist, tZ, dim_nu_Z, rangesXXList, rangesZZ_ij, rangesZXList)
            loss = make_loss2(tXlist,tZo, tZ, tnu_Z, dim_nu_Zo, rangesZoXList, rangesZoZo_ij, rangesZoZn_ij)
            print("time for making loss 2 is " + str(time.time() - temptime))
            p0 = tnu_Zo.pow(0.5).flatten().clone().requires_grad_(True)

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
            L = loss(p0)
            print("error is ", L.detach().cpu().numpy())
            #print("relative error loss", L.detach().cpu().numpy()/c.detach().cpu().numpy())
            lossTrack.append(L.detach().cpu().numpy())
            #tot.backward()
            return L

        for i in range(nb_iter):
            print("it ", i, ": ", end="")
            print(torch.cuda.memory_allocated(0))
            GPUtil.showUtilization()
            temptime = time.time()
            optimizer.step(closure)
            print("time to take a step is " + str(time.time() - temptime))
            #torch.cuda.empty_cache()

        if flag == 'all':
            tnZo = p0[0:3*len_Zo].detach().view(-1,3)
            tnnu_Zo = p0[3*len_Zo::].detach().view(-1,dim_nu_Zo)**2
        else:
            tnZo = tZo
            tnnu_Zo = p0.detach().view(-1,dim_nu_Zo)**2
        return tnZo, tnnu_Zo

    print("Starting Optim")
    print("sum tnu_Z before", tnu_Zo.sum())
    tZo, tnu_Zo = optimize(tZo, tnu_Zo, nb_iter = nb_iter0, flag = '')
    tnZo, tnnu_Zo = tZo, tnu_Zo
    #torch.cuda.empty_cache()
    tnZo, tnnu_Zo = optimize(tZo, tnu_Zo, nb_iter = nb_iter1, flag = 'all')
    print("sum tnnu_Z after", tnnu_Zo.sum())

    nZ = tnZo.detach().cpu().numpy()
    nnu_Z = tnnu_Zo.detach().cpu().numpy()
    
    # determine amount by which particles have moved and by which distributions have changed (squared error distance)
    distMove = np.sqrt(np.sum((Zopt - nZ)**2,axis=-1))
    distAlt = np.sqrt(np.sum((onuZopt-nnu_Z)**2,axis=-1))
    fig,ax = plt.subplots(2,1)
    ax[0].hist(distMove)
    ax[1].hist(distAlt)
    ax[0].set_title('Distance Moved')
    ax[1].set_title('Distribution Changed')
    fig.savefig(outpath+'_optimalZnu_ZwC' + str(C) + '_sig' + str(sig) + '_Nmax' + str(Nmax) + '_Npart' + str(Npart) + '_distancesStitch.png',dpi=300)
    np.savez(outpath+'_optimalZnu_ZwC' + str(C) + '_sig' + str(sig) + '_Nmax' + str(Nmax) + '_Npart' + str(Npart) + '_distancesStitch.npz',Z=Zopt,nZ=nZ,nu_Z=onuZopt,nnu_Z=nnu_Z,distMove=distMove,distAlt=distAlt)
    
    nnu_Z = reverseOneHot(nnu_Z,indsToKeep,maxV)
    print("nnu_Z shape should be number of particles by maxV")
    print(nnu_Z.shape)
    
    nZcomb = np.vstack((nZ,Z,Zkeep))
    print("should be equal ")
    print(str(nZcomb.shape[0]) + " vs . " + str(Z1.shape[0] + Z2.shape[0]))
    nuZcomb = np.vstack((nnu_Z,nuZ,nuZkeep))
    print("nu Z shape is " + str(nuZcomb.shape))
    nZedit = np.vstack((np.ones((nZ.shape[0],1)),np.zeros((Z.shape[0],1)),np.zeros((Zkeep.shape[0],1))))
    
    maxInd = np.argmax(nuZcomb,axis=-1)+1
    vtf.writeVTK(nZcomb,[maxInd,np.sum(nuZcomb,axis=-1),nZedit],['MAX_VAL_NU','TOTAL_MASS','OPTIMIZED'],outpath+'_optimalZnu_ZAllwC' + str(C) + '_sig' + str(sig) + '_Nmax' + str(Nmax) + '_Npart' + str(Npart) +'.vtk',polyData=None)
    fZ = outpath+'_optimalZnu_ZAllwC' + str(C) + '_sig' + str(sig) + '_Nmax' + str(Nmax) + '_Npart' + str(Npart) +'.npz'
    np.savez(fZ,Z=nZcomb, nu_Z=nuZcomb,Zo=nZ,nu_Zo=nnu_Z)

    # Not good !!!! -- removes low mass particles 
    mean = np.mean(nnu_Z.sum(axis=1))
    c = nnu_Z.sum(axis=1)>0.1*mean
    print("number of orig particles " + str(nnu_Z.shape[0]))
    print("number of new particles after remove low mass " + str(np.sum(c)))
    nZ = nZ[c,:]
    nnu_Z = nnu_Z[c,:]
    
    nZcomb = np.vstack((nZ,Z,Zkeep))
    nuZcomb = np.vstack((nnu_Z,nuZ,nuZkeep))
    nZedit = np.vstack((np.ones((nZ.shape[0],1)),np.zeros((Z.shape[0],1)),np.zeros((Zkeep.shape[0],1))))

    # Display of the result (in reduced integer set)
    maxInd = np.argmax(nuZcomb,axis=-1)+1
    vtf.writeVTK(nZcomb,[maxInd,np.sum(nuZcomb,axis=-1),nZedit],['MAX_VAL_NU','TOTAL_MASS','OPTIMIZED'],outpath+'_optimalZnu_ZwC' + str(C) + '_sig' + str(sig) + '_Nmax' + str(Nmax) + '_Npart' + str(Npart) +'.vtk',polyData=None)
    np.savez(outpath+'_optimalZnu_ZwC' + str(C) + '_sig' + str(sig) + '_Nmax' + str(Nmax) + '_Npart' + str(Npart) +'.npz',Z=nZcomb, nu_Z=nuZcomb,Zo=nZ,nu_Zo=nnu_Z)
    
    fX = outpath+'_XnuX.npz'
    X = np.vstack((X,Xkeep))
    nuX = np.vstack((nuX,nuXkeep))
    np.savez(fX,X=X, nu_X=nuX)

    # remove to see if release memory 
    f,ax = plt.subplots()
    ax.plot(np.arange(len(lossTrack)),np.asarray(lossTrack))
    ax.set_xlabel('iterations')
    ax.set_ylabel('cost')
    f.savefig(outpath+ '_optimalZcost_sig' + str(sig) + '_C' + str(C) + '_Nmax' + str(Nmax) + '_Npart' + str(Npart) + '.png',dpi=300)
    
    return X, nuX, nZcomb, nuZcomb, fX, fZ


    
def stitchAllQuadrants(inpathX,inpathZ,prefix,sigma,marginI,outpath,nb_iter0,nb_iter1,NmaxI,NpartI,maxVal):
    '''
    Example prefix is 0-100
    '''
    print("searching:")
    print(inpathX + '*' + prefix + '*XnuX._*npz')
    allX = glob.glob(inpathX + '*' + prefix + '*XnuX._*npz')
    print(len(allX))
    allZ = glob.glob(inpathZ + '*' + prefix + '*_ZnuZ._*' + '*ZAll*npz')
    print(len(allZ))
        
    allX.sort()
    allZ.sort()
    
    print(allX[0])
    print(allZ[0])
    print(allX[1])
    print(allZ[1])
    print(outpath + 'Sub_' + prefix + '_01')
    print(sigma)
    print(nb_iter0)
    print(nb_iter1)
    print(marginI)
    print(NmaxI)
    print(NpartI)
    print(maxVal)

    
    # stitch 0,1 and 2,3
    fX01,fZ01 = stitchQuadrants(allX[0],allX[1],allZ[0],allZ[1],outpath + 'Sub_' + prefix + '_01',sigma,nb_iter0, nb_iter1, margin=marginI,Nmax=NmaxI,Npart=NpartI,maxV=maxVal,optMethod='LBFGS')
    fX23,fZ23 = stitchQuadrants(allX[2],allX[3],allZ[2],allZ[3],outpath + 'Sub_' + prefix + '_23',sigma,nb_iter0, nb_iter1, margin=marginI,Nmax=NmaxI,Npart=NpartI,maxV=maxVal,optMethod='LBFGS')

    #torch.cuda.empty_cache()
    #cuda.current_context().deallocations.clear()

    fX,fZ = stitchQuadrants(fX01,fX23,fZ01,fZ23,outpath + 'Sub_' + prefix + '_0123',sigma,nb_iter0,nb_iter1,margin=marginI, Nmax=NmaxI, Npart=NpartI,maxV=maxVal,optMethod='LBFGS')
    print("wrote " + fX + ", " + fZ)
    
    return

def stitchAllSlabs(inpathX,inpathZ,prefix,sigma,marginI,outpath,nb_iter0,nb_iter1,NmaxI,NpartI,maxVal,ax=2):
    '''
    Example prefix is _0123_ with inpathX = /cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um/Experiments/Stitched/Sub
    '''
    
    if len(inpathX) > 1:
        allX = inpathX
        allZ = inpathZ
    else:
    
        allX = glob.glob(inpathX[0] + '*' + prefix + 'XnuX.npz') # assume ordered numerically 
        allZ = glob.glob(inpathZ[0] + '*' + prefix + '*optimalZnu_ZAll*0.npz')
    
        allX.sort()
        allZ.sort()
    
    print("number of X files is " + str(len(allX)))
    print("number of Z files is " + str(len(allZ)))
    print(allX)
    print(allZ)
    
    slabsX = allX
    slabsZ = allZ
    
    pref = 2
    # temporary fix
    '''
    pref = 4
    
    slabsX = ['/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um/ZApprox-XComb_sig0.1/Sub_2slabs_0-1_XnuX.npz','/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um/ZApprox-XComb_sig0.1/Sub_2slabs_2-3_XnuX.npz','/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um/ZApprox-XComb_sig0.1/Sub_2slabs_4-5_XnuX.npz','/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um/ZApprox-XComb_sig0.1/Sub_2slabs_6-7_XnuX.npz','/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um/ZApprox-XComb_sig0.1/Sub_2slabs_8-9_XnuX.npz', '/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um/ZApprox-XComb_sig0.1/Sub_2slabs_10-11_XnuX.npz', '/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um/ZApprox-XComb_sig0.1/Sub__1200-1300_0123_XnuX.npz']
    slabsZ = ['/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um/ZApprox-XComb_sig0.1/Sub_2slabs_0-1_optimalZnu_ZAllwC1.2_sig0.1_Nmax5000.0_Npart1000.0.npz', '/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um/ZApprox-XComb_sig0.1/Sub_2slabs_2-3_optimalZnu_ZAllwC1.2_sig0.1_Nmax5000.0_Npart1000.0.npz', '/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um/ZApprox-XComb_sig0.1/Sub_2slabs_4-5_optimalZnu_ZAllwC1.2_sig0.1_Nmax5000.0_Npart1000.0.npz', '/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um/ZApprox-XComb_sig0.1/Sub_2slabs_6-7_optimalZnu_ZAllwC1.2_sig0.1_Nmax5000.0_Npart1000.0.npz', '/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um/ZApprox-XComb_sig0.1/Sub_2slabs_8-9_optimalZnu_ZAllwC1.2_sig0.1_Nmax5000.0_Npart1000.0.npz', '/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um/ZApprox-XComb_sig0.1/Sub_2slabs_10-11_optimalZnu_ZAllwC1.2_sig0.1_Nmax5000.0_Npart1000.0.npz', '/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um/ZApprox-XComb_sig0.1/Sub__1200-1300_0123_optimalZnu_ZAllwC1.2_sig0.1_Nmax5000.0_Npart1000.0.npz']
    '''
    while len(slabsX) > 1:
        tempX = []
        tempZ = []
        for i in range(0,len(slabsX)-1,2):
            print("combining " + slabsZ[i] + ", " + slabsZ[i+1])
            combX01,combnuX01,combZ01,combnuZ01,fX01,fZ01 = stitchSlabs(slabsX[i],slabsX[i+1],slabsZ[i],slabsZ[i+1],outpath + 'Sub_' + str(pref) + 'slabs_' + str(i) + '-' + str(i+1),sigma,nb_iter0, nb_iter1, margin=marginI,Nmax=NmaxI,Npart=NpartI,maxV=maxVal,optMethod='LBFGS',ax=ax)
            print("wrote " + fX01 + ", " + fZ01)
            tempX.append(fX01)
            tempZ.append(fZ01)
        if 2*len(tempX) < len(slabsX):
            tempX.append(slabsX[len(slabsX)-1])
            tempZ.append(slabsZ[len(slabsZ)-1])
        slabsX = tempX
        slabsZ = tempZ
        pref = pref*2
    return
    
def stitchTwoSlabs(inpathX,inpathZ,prefix,sigma,marginI,outpath,nb_iter0,nb_iter1,NmaxI,NpartI,maxVal,ax=2):
    '''
    Example prefix is _0123_ with inpathX = /cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um/Experiments/Stitched/Sub
    '''
    pref = 2
    slabsX = ['/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um/ZApprox-XComb_sig0.1_new/Sub__0600-0700_0123_XnuX.npz','/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um/ZApprox-XComb_sig0.1_new/Sub__0700-0800_0123_XnuX.npz', '/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um/ZApprox-XComb_sig0.1_new/Sub__0800-0900_0123_XnuX.npz', '/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um/ZApprox-XComb_sig0.1_new/Sub__0900-1000_0123_XnuX.npz', '/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um/ZApprox-XComb_sig0.1_new/Sub__1000-1100_0123_XnuX.npz', '/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um/ZApprox-XComb_sig0.1_new/Sub__1100-1200_0123_XnuX.npz']
    slabsZ = ['/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um/ZApprox-XComb_sig0.1_new/Sub__0600-0700_0123_optimalZnu_ZAllwC1.2_sig0.1_Nmax5000.0_Npart1000.0.npz','/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um/ZApprox-XComb_sig0.1_new/Sub__0700-0800_0123_optimalZnu_ZAllwC1.2_sig0.1_Nmax5000.0_Npart1000.0.npz', '/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um/ZApprox-XComb_sig0.1_new/Sub__0800-0900_0123_optimalZnu_ZAllwC1.2_sig0.1_Nmax5000.0_Npart1000.0.npz', '/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um/ZApprox-XComb_sig0.1_new/Sub__0900-1000_0123_optimalZnu_ZAllwC1.2_sig0.1_Nmax5000.0_Npart1000.0.npz', '/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um/ZApprox-XComb_sig0.1_new/Sub__1000-1100_0123_optimalZnu_ZAllwC1.2_sig0.1_Nmax5000.0_Npart1000.0.npz', '/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um/ZApprox-XComb_sig0.1_new/Sub__1100-1200_0123_optimalZnu_ZAllwC1.2_sig0.1_Nmax5000.0_Npart1000.0.npz']
    tempX = []
    tempZ = []
    for i in range(0,len(slabsX)-1,2):
        combX01,combnuX01,combZ01,combnuZ01,fX01,fZ01 = stitchSlabs(slabsX[i],slabsX[i+1],slabsZ[i],slabsZ[i+1],outpath + 'Sub_' + str(pref) + 'slabs_' + str(i+8) + '-' + str(i+9),sigma,nb_iter0, nb_iter1, margin=marginI,Nmax=NmaxI,Npart=NpartI,maxV=maxVal,optMethod='LBFGS',ax=ax)
        print("wrote " + fX01 + ", " + fZ01)
        tempX.append(fX01)
        tempZ.append(fZ01)
    if 2*len(tempX) < len(slabsX):
        tempX.append(slabsX[len(slabsX)-1])
        tempZ.append(slabsZ[len(slabsZ)-1])
    slabsX = tempX
    slabsZ = tempZ
    pref = pref*2
    return


        
            
 