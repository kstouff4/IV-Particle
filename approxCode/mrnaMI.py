#import ntpath
#from numba import jit, prange, int64
import time
import sys
from sys import path as sys_path
sys_path.append('/cis/home/kstouff4/Documents/SurfaceTools/')
#sys_path.append('/cis/home/kstouff4/Documents/MeshRegistration/master/py-lddmm/')
#sys_path.append('/cis/home/kstouff4/Documents/MeshRegistration/master/py-lddmm/base')
import vtkFunctions as vtf
#import multiprocessing as mp
#from multiprocessing import Pool

#import torch
import numpy as np
import numpy.matlib

import os, psutil
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd

import glob
import scipy as sp
from scipy import signal
##############################################################
# Compute mRNA MI for detected transcripts

def singleMI(detectedTransCSV,cSize,mSize,k=4,meta=None,feat=None,makeOneHot=True):
    '''
    read in detected transcripts: global_x = 2, global_y = 3, global_z = 4, geneName = 8
    cSize = cube size (for building histograms)
    mSize = megaCube size (for doing convolution)
    k = number of histogram bins
    '''
    if ('npz' in detectedTransCSV):
        info = np.load(detectedTransCSV) # assume in the form of X and nu_X
        coords = info[info.files[0]][:,0:2] # only take x and y
        if feat is None:
            feat = 'nu_X'
        if makeOneHot:
            ugenes,inv = np.unique(info[feat],return_inverse=True)
            invOneHot = np.zeros((inv.shape[0],len(ugenes)), dtype=np.bool8)
            invOneHot[np.arange(inv.shape[0]),inv] = 1
        else:
            invOneHot = info[feat] # do not convert to one-hot (assume already in this)
            inv = np.argmax(info[feat],axis=-1)
    # 3D Allen MERFISH Data
    elif meta is None:
        df = pd.read_csv(detectedTransCSV)
        print('done')
        x_ = df['global_x'].to_numpy() # units of micron for allen 3D
        y_ = df['global_y'].to_numpy()
        z_ = df['global_z'].to_numpy()
        z_ = z_*(25.0/np.max(z_)) # scale to be in microns based on 25 micron unit; assume evenly spaced

        coords = np.stack((x_,y_),axis=-1) # try 2D first
        ugenes, inv = np.unique(df['gene'], return_inverse=True) # inverse = occurrences of elements 
        print("unique num of genes is " + str(len(ugenes)))
        coordsTot = np.stack((x_,y_,z_),axis=-1)
        np.savez(detectedTransCSV.replace('.csv','.npz'),coordsTot=coordsTot,geneInd=inv,geneList=ugenes) # save list of coords by gene
        fig,ax = plt.subplots()
        im = ax.scatter(coords[:,0],coords[:,1],s=1,c=inv)
        fig.colorbar(im,ax=ax)
        fig.savefig(detectedTransCSV.replace('.csv','.png'),dpi=300)
    
        invOneHot = np.zeros((inv.shape[0],len(ugenes)), dtype=np.bool8)
        invOneHot[np.arange(inv.shape[0]),inv] = 1 # number of detections by number of counts 
    else:
        df = pd.read_csv(meta)
        x_ = df['center_x'].to_numpy()
        y_ = df['center_y'].to_numpy()
        coords = np.stack((x_,y_),axis=-1)
        df2 = pd.read_csv(detectedTransCSV)
        invOneHot = df2.iloc[:,1:].values.astype('float32')
        ugenes = df2.columns[1:]
        np.savez(detectedTransCSV.replace('.csv','.npz'),coords=coords,geneCounts=invOneHot,geneList=ugenes)
        
    
    # divide mRNA coords and features into cubes based on location and given size 
    
    coords_labels = np.floor((coords - np.floor(np.min(coords,axis=0)))/cSize).astype(int) # minimum number of cubes in x and y 
    totCubes = (np.max(coords_labels[:,0])+1)*(np.max(coords_labels[:,1])+1)
    xC = np.arange(np.max(coords_labels[:,0])+1)*cSize + np.floor(np.min(coords[:,0])) + cSize/2.0
    yC = np.arange(np.max(coords_labels[:,1])+1)*cSize + np.floor(np.min(coords[:,1])) + cSize/2.0
    XC,YC = np.meshgrid(xC,yC,indexing='ij')
    cubes_centroids = np.stack((XC,YC),axis=-1)
    cubes_indices = np.reshape(np.arange(totCubes),(cubes_centroids.shape[0],cubes_centroids.shape[1]))
    coords_labels_tot = cubes_indices[coords_labels[:,0],coords_labels[:,1]]
    print("coords_labels_tot shape is (probably unraveled list)" )
    print(coords_labels_tot.shape)
    
    '''
    coords_labels = grid_cluster(coords, cSize)
    coords_ranges, cubes_centroids, _ = cluster_ranges_centroids(coords, coords_labels)
    tcoords = torch.tensor(coords).type(dtype)
    tMRNA = torch.tensor(invOneHot).type(dtype)
    tcoords, labelsSort = sort_clusters(tcoords, coords_labels) # sorting the labels
    tMRNA, labelsSort = sort_cluters(tMRNA, coords_labels)
    '''
    
    # make counts per cube
    if meta is None:
        cubes_mrna = np.zeros((totCubes,invOneHot.shape[1]))
        print("number of unique cubes to which mrna belong to")
        print(len(np.unique(coords_labels_tot)))
        d = np.asarray([coords_labels_tot.astype(int),inv.astype(int)]) # labels of cube in first column, index of gene
        d = d.T
        print("d shape")
        print(d.shape)
        u,co = np.unique(d,return_counts=True,axis=0) # number of unique cube, gene pairs 
        print(u.shape)
        print(co.shape)
        cubes_mrna[u[:,0],u[:,1]] = co # cube, gene index with number of counts 
    else:
        cubes_mrna = np.zeros((totCubes,invOneHot.shape[1]))
        for l in np.unique(coords_labels_tot):
            cubes_mrna[l,:] = np.sum(invOneHot[coords_labels_tot == l,:],axis=0)
    '''
    for c in range(totCubes):
        if np.sum(coords_labels_tot == c) > 0:
            i = invOneHot[coords_labels_tot == c,:]
            cubes_mrna[c,:] = np.sum(i,axis=0)
    '''
    # find histograms for counts per cube
    cubes_binned = np.zeros_like(cubes_mrna)
    qSize = 1.0/k
    qt = np.quantile(cubes_mrna,qSize,axis=0)
    print("first quantile is")
    print(qt)
    print("q size should be 0.25")
    print(qSize)
    cubes_binned[cubes_mrna <= qt] = 1
    q = qSize*2
    count = 2
    while (q < 1.0):
        qt = np.quantile(cubes_mrna,q,axis=0)
        print("quantile is")
        print(qt)
        cubes_binned[(cubes_binned == 0)*(cubes_mrna <= qt)] = count
        q = q + qSize
        count = count + 1
    qt = np.quantile(cubes_mrna,q,axis=0)
    print("quantile is")
    print(qt)
    cubes_binned[(cubes_binned == 0)*(cubes_mrna <= qt)] = count
    print("should be 1 to " + str(k))
    print(np.unique(cubes_binned))
    
    bins=np.reshape(cubes_binned,(cubes_centroids.shape[0],cubes_centroids.shape[1],cubes_binned.shape[-1]))
    
    # write out bins per individual cubes as particles
    if ('csv' in detectedTransCSV):
        np.savez(detectedTransCSV.replace('.csv','2DcubeMRNA_cSize' + str(cSize) + '_k' + str(k) + '.npz'),cubes_binned=cubes_binned, cubes_centroids=cubes_centroids,bins=bins,cubes_mrna=cubes_mrna)
    else:
        np.savez(detectedTransCSV.replace('.npz','2DcubeMRNA_cSize' + str(cSize) + '_k' + str(k) + '.npz'),cubes_binned=cubes_binned, cubes_centroids=cubes_centroids,bins=bins,cubes_mrna=cubes_mrna)
        
        
    f,ax = plt.subplots(2,1)
    ccr0 = np.ravel(cubes_centroids[...,0])
    ccr1 = np.ravel(cubes_centroids[...,1])
    ax[0].scatter(ccr0,ccr1)
    ax[0].set_title("All Cubes")
    indsFill = np.ravel(np.sum(cubes_mrna,axis=-1)) > 0
    ax[1].scatter(ccr0[indsFill],ccr1[indsFill])
    ax[1].set_title("Non-empty Cubes: " + str(np.sum(indsFill)))
    if ('csv' in detectedTransCSV):
        f.savefig(detectedTransCSV.replace('.csv','centroidsLocs_cSize' + str(cSize) + '.png'),dpi=300)
    else:
        f.savefig(detectedTransCSV.replace('.npz','centroidsLocs_cSize' + str(cSize) + '.png'),dpi=300)
        
    return
    
def convolveHalfPlane(cubeFile,mSize,axC=0):
    '''
    cubeFile should have cube centers (cube centroids) and cubes binned (list) and bins (arranged as cube_centroids) information
    mSize should be even number (cubic window of megacube dictating convolution)
    ax = indicates whether to split vertically (ax = 0) or horizontally (ax = 1)
    '''
    info = np.load(cubeFile)
    bins = info['bins']
    print("size of bins: ", bins.shape)
    quants = len(np.unique(bins))
    mWin0 = np.zeros((mSize,mSize,bins.shape[-1]))
    mWin1 = np.zeros((mSize,mSize,bins.shape[-1]))
    mSizeCubes = mSize**2
    if (axC == 0):
        mWin0[0:int(mSize/2),...] = 1.0/mSizeCubes
        mWin1[int(mSize/2):,...] = 1.0/mSizeCubes
    elif (axC == 1):
        mWin0[:,0:int(mSize/2),...] = 1.0/mSizeCubes
        mWin1[:,int(mSize/2):,...] = 1.0/mSizeCubes
    Px0 = []
    Px1 = []
    Px = []
    
    Pm0 = np.zeros_like(bins)
    Pm1 = np.zeros_like(bins)
    
    for q in range(1,quants+1):
        Px0.append(sp.signal.fftconvolve(bins == q,mWin0,mode='same',axes=(0,1)))
        Px1.append(sp.signal.fftconvolve(bins == q,mWin1,mode='same',axes=(0,1)))
    
    for q in range(quants):
        Px.append(Px0[q] + Px1[q])
        Pm0 += Px0[q]
        Pm1 += Px1[q]
    
    MI = np.zeros_like(bins)
    for q in range(quants):
        r0 = np.reciprocal((Px[q]*Pm0),where=(Px[q]*Pm0 > 0))
        r1 = np.reciprocal((Px[q]*Pm1),where=(Px[q]*Pm1 > 0))
        
        MI += Px0[q]*np.log(Px0[q]*r0,where=(Px0[q]*r0 > 0))
        MI += Px1[q]*np.log(Px1[q]*r1,where=(Px1[q]*r1 > 0))
    
    J = np.sum(MI,axis=(0,1))
    print("size of J: ", J.shape)
    np.savez(cubeFile.replace('.npz','_msize' + str(mSize) + '_ax' + str(axC) + '_MI.npz'),J=J,MI=MI)
    f,ax = plt.subplots(2,1)
    ax[0].bar(np.arange(bins.shape[-1]),np.squeeze(J))
    ax[1].hist(np.squeeze(J))
    f.savefig(cubeFile.replace('.npz','_msize' + str(mSize) + '_ax' + str(axC) + '_MIhisto.png'),dpi=300)
    
    mx = np.argmax(J)
    f,ax = plt.subplots()
    im = ax.imshow(MI[...,mx])
    f.colorbar(im,ax=ax)
    f.savefig(cubeFile.replace('.npz','_msize' + str(mSize) + '_ax' + str(axC) + '_MImaxMI_ind' + str(mx) + '.png'),dpi=300)
    
    mx = np.argmin(J)
    f,ax = plt.subplots()
    im = ax.imshow(MI[...,mx])
    f.colorbar(im,ax=ax)
    f.savefig(cubeFile.replace('.npz','_msize' + str(mSize) + '_ax' + str(axC) + '_MIminMI_ind' + str(mx) + '.png'),dpi=300)

    '''
    
    y = np.where(J > np.quantile(J,0.99)) # about 10 genes
    y = y[0]
    for i in range(len(y)):
        f,ax = plt.subplots(2,1)
        im0 = ax[0].imshow(MI[...,y[i]])
        f.colorbar(im0,ax=ax[0])
        im1 = ax[1].imshow(np.log(MI[...,y[i]]+1)/np.log(10))
        f.colorbar(im1,ax=ax[1])
        f.savefig(cubeFile.replace('.npz','_msize' + str(mSize) + '_ax' + str(axC) + '_MImaxMI_ind' + str(y[i]) + '_withLog.png'),dpi=300)
    '''
    
    # try removing boundary pixels
    bou = int(mSize/2)
    MI[0:bou,...] = 0
    MI[-bou:,...] = 0
    MI[:,0:bou,...] = 0
    MI[:,-bou:,...] = 0
    
    J0 = np.sum(MI,axis=(0,1))
    np.savez(cubeFile.replace('.npz','_msize' + str(mSize) + '_ax' + str(axC) + '_MI_nobound.npz'),J=J0,MI=MI)
    f,ax = plt.subplots(2,1)
    ax[0].bar(np.arange(bins.shape[-1]),np.squeeze(J0))
    ax[1].hist(np.squeeze(J0))
    f.savefig(cubeFile.replace('.npz','_msize' + str(mSize) + '_ax' + str(axC) + '_MIhisto_nobound.png'),dpi=300)
    
    mx = np.argmax(J0)
    f,ax = plt.subplots()
    im = ax.imshow(MI[...,mx])
    f.colorbar(im,ax=ax)
    f.savefig(cubeFile.replace('.npz','_msize' + str(mSize) + '_ax' + str(axC) + '_MImaxMI_ind' + str(mx) + '_nobound.png'),dpi=300)
    
    mx = np.argmin(J0)
    f,ax = plt.subplots()
    im = ax.imshow(MI[...,mx])
    f.colorbar(im,ax=ax)
    f.savefig(cubeFile.replace('.npz','_msize' + str(mSize) + '_ax' + str(axC) + '_MIminMI_ind' + str(mx) + '_nobound.png'),dpi=300)

    '''
    y = np.where(J0 > np.quantile(J0,0.99)) # about 10 genes
    y = y[0]
    for i in range(len(y)):
        f,ax = plt.subplots(2,1)
        im0 = ax[0].imshow(MI[...,y[i]])
        f.colorbar(im0,ax=ax[0])
        im1 = ax[1].imshow(np.log(MI[...,y[i]]+1)/np.log(10))
        f.colorbar(im1,ax=ax[1])
        f.savefig(cubeFile.replace('.npz','_msize' + str(mSize) + '_ax' + str(axC) + '_MImaxMI_ind' + str(y[i]) + '_withLog_nobound.png'),dpi=300)
    '''

    return

def wholeBrainMI(dirName,saveName,featNameMat=None):
    '''
    Give directory where you have stored the cube counts (e.g. /cis/home/kstouff4/Documents/SpatialTranscriptomics/Mouse/Mouse1_20220506/zipfiles1/*/)
    '''
    ax0_fils = glob.glob(dirName + '*_ax0_MI_nobound.npz')
    ax0_fils.sort()

    ax1_fils = glob.glob(dirName + '*_ax1_MI_nobound.npz')
    ax1_fils.sort()

    numFils = len(ax0_fils)
    ax0f = np.load(ax0_fils[0])
    ax1f = np.load(ax1_fils[0])
    J0 = ax0f['J']
    J1 = ax1f['J']
    J = J0+J1
    Jtot = J
    Js = []
    slNames = []
    Js.append(J)
    indSort = np.argsort(J)
    p = ax0_fils[0].split('/')[-2]
    slNames.append(p)
    f,ax = plt.subplots(figsize=(16,8))
    ax.bar(np.arange(len(J)),J[indSort])
    ax.set_xticks(list(np.arange(len(J))))
    ax.set_xticklabels(list(indSort),rotation='vertical',fontsize='x-small')
    f.savefig(saveName + p + '_MI_nobound_histo.png',dpi=300)
    
    for i in range(1,numFils):
        ax0f = np.load(ax0_fils[i])
        ax1f = np.load(ax1_fils[i])
        J0 = ax0f['J']
        J1 = ax1f['J']
        J = J0+J1
        Js.append(J)
        indSort = np.argsort(J)
        p = ax0_fils[i].split('/')[-2]
        f,ax = plt.subplots(figsize=(16,8))
        ax.bar(np.arange(len(J)),J[indSort])
        ax.set_xticks(list(np.arange(len(J))))
        ax.set_xticklabels(list(indSort),rotation='vertical',fontsize='x-small')
        f.savefig(saveName + p+'_MI_nobound_histo.png',dpi=300)
        slNames.append(p)
        if (J.shape != Jtot.shape):
            print("shapes mismatch")
            continue
        else:
            Jtot += J
        

    np.savez(saveName+'all_ax01_MI_nobound.npz',Js=Js,Jtot=Jtot,slNames=slNames)
    indSort = np.argsort(Jtot)
    f,ax = plt.subplots(figsize=(16,8))
    ax.bar(np.arange(len(Jtot)),Jtot[indSort])
    ax.set_xticklabels(list(indSort))
    f.savefig(saveName+'all_MI_nobound_histo.png',dpi=300)

    return
     
    # group cubes into sliding window mega cubes and for each megacube, compute the P_x,m = number of cubes with m (histo bin) = m and x = half of cube (e.g. top or bottom, left or right) (FT the filter)
    
    # compute I_c(X,M) = \sum_{x,m} P_x,m log P_x,m / P_x*P_m
    
    
    
    