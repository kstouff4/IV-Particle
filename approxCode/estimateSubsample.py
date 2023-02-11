import ntpath
from numba import jit, prange, int64
from sys import path as sys_path
sys_path.append('/cis/home/kstouff4/Documents/MeshRegistration/MeshLDDMMQP/master-KMS/py-lddmm/')
sys_path.append('/cis/home/kstouff4/Documents/MeshRegistration/MeshLDDMMQP/master-KMS/py-lddmm/base')
sys_path.append('/cis/home/kstouff4/Documents/SurfaceTools/')
import vtkFunctions as vtf
import os
from base import loggingUtils
import multiprocessing as mp
from multiprocessing import Pool
import glob
import pandas as pd
import matplotlib
if 'DISPLAY' in os.environ:
    matplotlib.use('qt5Agg')
else:
    matplotlib.use("Agg")
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import matplotlib.pyplot as plt
import numpy as np
import imageio.v3 as iio
import cv2
import torch
import numpy.matlib

import pykeops
import socket
pykeops.set_build_folder("~/.cache/keops"+pykeops.__version__ + "_" + (socket.gethostname()))

from pykeops.torch import Vi, Vj
from pykeops.torch import LazyTensor
from skimage.segmentation import watershed
from base.meshes import buildImageFromFullListHR, buildMeshFromCentersCounts, buildMeshFromImageData
from PIL import Image
Image.MAX_IMAGE_PIXELS
Image.MAX_IMAGE_PIXELS=1e10 # forget attack
import nibabel as nib
from qpsolvers import solve_qp
from base.kernelFunctions import Kernel, kernelMatrix
import logging
from base.curveExamples import Circle
from base.surfaces import Surface
from base.surfaceExamples import Sphere, Rectangle
from base import loggingUtils
from base.meshes import Mesh, buildMeshFromMerfishData
from base.affineRegistration import rigidRegistration, rigidRegistration_varifold
from base.meshMatching import MeshMatching, MeshMatchingParam
from base.mesh_distances import varifoldNormDef

from datetime import datetime
import time

import scipy as sp
from scipy import sparse
from scipy.sparse import coo_array
#import tensorflow as tf

########################################################################


def make_target3DImage(imgFile,ax=2,indS=0,indF=-1,res=0.01):
    im = nib.load(imgFile)
    imageO = np.asanyarray(im.dataobj).astype('float32')
    if (ax == 0):
        image = imageO[indS:indF,...]
    elif (ax == 1):
        image = imageO[:,indS:indF,...]
    elif (ax == 2):
        image = imageO[:,:,indS:indF,...]
    else:
        image = imageO
    
    #image = image[300:500,300:500,...]
    
    x0 = np.arange(image.shape[0])*res
    x1 = np.arange(image.shape[1])*res
    x2 = np.arange(image.shape[2])*res
    
    x0 -= np.mean(x0)
    x1 -= np.mean(x1)
    x2 -= np.mean(x2)
    
    X0, X1, X2 = np.meshgrid(x0,x1,x2,indexing='ij')
    X = np.stack((X0,X1,X2),axis=-1)
    
    uniqueVals, imageRZ = np.unique(image,return_inverse=True)
    nu_X = np.zeros((image.shape[0],image.shape[1],image.shape[2],len(uniqueVals)-1))
    for i in range(1,len(uniqueVals)):
        nu_X[...,i-1] = np.squeeze(image == uniqueVals[i])
    print("confirming have probability distribution (should all be 1)")
    print(np.sum(nu_X,axis=-1))
    print(np.sum(np.sum(nu_X,axis=-1) != 1))
    
    X = np.reshape(X,(X.shape[0]*X.shape[1]*X.shape[2], X.shape[-1]))
    nu_X = np.reshape(nu_X,(nu_X.shape[0]*nu_X.shape[1]*nu_X.shape[2],nu_X.shape[-1]))
    print("nu_X shape ")
    print(nu_X.shape)
    
    vol = image.shape[0]*res*image.shape[1]*res*image.shape[2]*res
    
    return X, nu_X, vol

def make_subsample(X, nu_X,  N, tot=None):
  # Input:
  # X is the initial data
  # N is the number of random subsample

  # Output 
  # Y is the output subsample

    sub_ind = np.random.choice(X.shape[0],replace = False, size = N)
    print(sub_ind.shape)
    Z = X[sub_ind,:]
    if (tot is None):
        nu_Z = nu_X[sub_ind,:]*X.shape[0]/N # gives less mass 
    else:
        nu_Z = np.zeros((N,tot))
        nu_Zs = nu_X[sub_ind,:]
        for i in range(N):
            nu_Z[i,int(nu_Zs[i]-1)] = 1
        nu_Z = nu_Z*X.shape[0]/N
        
    return Z, nu_Z

def makeAllXandZ(imgFile, outpath, thickness=100, res=0.01,sig=0.1,C=1,flip=False):
    '''
    Loop through sections of 100 (given by thickness) slices at a time (1 mm of tissue);
    assume taking slabs along z-axis
    '''
    
    filesWritten = []
    
    # load image 
    im = nib.load(imgFile)
    imageO = np.asanyarray(im.dataobj).astype('float32')
    
    x0 = np.arange(imageO.shape[0])*res
    x1 = np.arange(imageO.shape[1])*res
    x2 = np.arange(imageO.shape[2])*res
    
    x0 -= np.mean(x0)
    x1 -= np.mean(x1)
    x2 -= np.mean(x2)
    
    uniqueVals = np.unique(imageO)
    numUniqueMinus0 = len(uniqueVals)-1
    
    if (flip):
        x2t = x0
        imageO = np.swapaxes(imageO,0,2)
        x0 = x2
        x2 = x2t
    
    z = 0
    if (thickness < 0):
        thickness = imageO.shape[2]-1
    while (z < imageO.shape[2]-thickness):
        X0, X1, X2 = np.meshgrid(x0,x1,x2[z:z+thickness],indexing='ij')
        X = np.stack((X0,X1,X2),axis=-1)
        image = imageO[:,:,z:z+thickness,...]
        #nu_X = np.zeros((image.shape[0],image.shape[1],image.shape[2],len(uniqueVals)-1)).astype('float32')
        nu_X = np.zeros((image.shape[0],image.shape[1],image.shape[2])).astype('float32')
        for i in range(1,len(uniqueVals)):
            #nu_X[...,i-1] = np.squeeze(image == uniqueVals[i])
            nu_X += np.squeeze(image == uniqueVals[i])*i
        X = np.reshape(X,(X.shape[0]*X.shape[1]*X.shape[2], X.shape[-1]))
        #nu_X = np.reshape(nu_X,(nu_X.shape[0]*nu_X.shape[1]*nu_X.shape[2],nu_X.shape[-1]))
        #tokeep = np.sum(nu_X,axis=-1) > 0 
        nu_X = np.ravel(nu_X)
        tokeep = nu_X > 0
        print("total points before")
        print(X.shape[0])
        X = X[tokeep,...]
        nu_X = nu_X[tokeep,...]
        if (len(nu_X.shape) < 2):
            nu_X = nu_X[...,None]
        print("total points now")
        print(X.shape[0])
        tot = len(np.unique(nu_X))
        print("total orig: " + str(len(uniqueVals)))
        print("total should be 1 less: " + str(tot))
        #maxIndnu = np.argmax(nu_X,axis=-1)+1
        #vtf.writeVTK(X,[maxIndnu],['MAX_VAL_NU'],outpath+'_' + str(z) + '-' + str(z+thickness) + '_XnuX.vtk',polyData=None)
        vtf.writeVTK(X,[nu_X],['NU_X'],outpath+'_' + str(z) + '-' + str(z+thickness) + '_XnuX.vtk',polyData=None)
        np.savez(outpath+'_' + str(z) + '-' + str(z+thickness) + '_XnuX.npz',X=X,nu_X=nu_X)
        filesWritten.append(outpath+'_' + str(z) + '-' + str(z+thickness) + '_XnuX.npz')
        
        if (C > 0):
            vol = image.shape[0]*res*image.shape[1]*res*image.shape[2]*res
            N = round(C*vol/(sig**3))
            Z,nu_Z = make_subsample(X,nu_X,N,numUniqueMinus0)
            maxIndnu = np.argmax(nu_Z,axis=-1)+1
            vtf.writeVTK(Z,[maxIndnu],['MAX_VAL_NU'],outpath+'_' + str(z) + '-' + str(z+thickness) + '_ZnuZ.vtk',polyData=None)

            # save Z and nu_Z and X and nu_X
            np.savez(outpath+'_' + str(z) + '-' + str(z+thickness) + '_ZnuZ.npz',Z=Z,nu_Z=nu_Z)
       
        z = z + thickness    
    return X,nu_X

def compileZ(pathToOutput,pref=None):
    if (pref is None):
        f = glob.glob(pathToOutput+'*Znu*.npz')
    else:
        f = glob.glob(pathToOutput +'*Znu*.npz')
        newF = []
        for g in f:
            if pref in g and 'dist' not in g:
                newF.append(g)
        f = newF
    print(f)
    o = np.load(f[0])
    print(o.files)
    oZ = o['Z']
    onuZ = o['nu_Z']
    for i in range(1,len(f)):
        o = np.load(f[i])
        oZ = np.vstack((oZ,o['Z']))
        onuZ = np.vstack((onuZ,o['nu_Z']))
    Z = oZ
    nu_Z = onuZ
    np.savez(pathToOutput+'Total_ZnuZ.npz',Z=Z,nu_Z=nu_Z)
    maxIndnu = np.argmax(nu_Z,axis=-1)+1
    vtf.writeVTK(Z,[maxIndnu],['MAX_VAL_NU'],pathToOutput+'Total_ZnuZ.vtk',polyData=None)
    return

def compileX(pathToOutput):
    f = glob.glob(pathToOutput+'*XnuX.npz')
    print(f)
    o = np.load(f[0])
    print(o.files)
    oZ = o['X']
    onuZ = o['nu_X']
    for i in range(1,len(f)):
        o = np.load(f[i])
        oZ = np.vstack((oZ,o['X']))
        onuZ = np.vstack((onuZ,o['nu_X']))
    Z = oZ
    nu_Z = onuZ
    np.savez(pathToOutput+'Total_XnuX.npz',X=Z,nu_X=nu_Z)
    maxIndnu = np.argmax(nu_Z,axis=-1)+1
    vtf.writeVTK(Z,[maxIndnu],['MAX_VAL_NU'],pathToOutput+'Total_XnuX.vtk',polyData=None)
    return

def splitParticles(particleFile,parts,ax0=True,ax1=True,ax2=True):
    '''
    split group of particles into # parts along each axis (same number of parts)
    '''
    particles = np.load(particleFile) # assume have X and nuX
    if ('X' in particles.files):
        oX = particles['X']
        onuX = particles['nu_X']
    elif ('Z' in particles.files):
        oX = particles['Z']
        onuX = particles['nu_Z']
    else:
        print("Didn't find X or Z")
    
    # get dim for each axis
    bounds = np.max(oX,axis=0) - np.min(oX,axis=0)
    minX = np.min(oX,axis=0)
    maxX = np.max(oX,axis=0)
    if (ax0):
        ax0oX = []
        ax0onuX = []
        for p in range(parts-1):
            lb = minX[0] + p*np.round(bounds[0]/parts)
            up = lb + np.round(bounds[0]/parts)
            inds = (oX[:,0] >= lb)*(oX[:,0] < up)
            ax0oX.append(oX[inds])
            ax0onuX.append(onuX[inds])
        lb = minX[0] + (parts-1)*np.round(bounds[0]/parts)
        inds = (oX[:,0] >= lb)
        ax0oX.append(oX[inds])
        ax0onuX.append(onuX[inds])
    else:
        ax0oX = oX
        ax0onuX = onuX
    if (ax1):
        ax1oX = []
        ax1onuX = []
        for a in range(len(ax0oX)):
            aoX = ax0oX[a]
            aonuX = ax0onuX[a]
            for p in range(parts-1):
                lb = minX[1] + p*np.round(bounds[1]/parts)
                up = lb + np.round(bounds[1]/parts)
                inds = (aoX[:,1] >= lb)*(aoX[:,1] < up)
                ax1oX.append(aoX[inds])
                ax1onuX.append(aonuX[inds])
            lb = minX[1] + (parts-1)*np.round(bounds[1]/parts)
            inds = (aoX[:,1] >= lb)
            ax1oX.append(aoX[inds])
            ax1onuX.append(aonuX[inds])
    else:
        ax1oX = ax0ox
        ax1onuX = ax0onuX
    if (ax2):
        ax2oX = []
        ax2onuX = []  
        for a in range(len(ax1oX)):
            aoX = ax1oX[a]
            aonuX = ax1onuX[a]
            for p in range(parts-1):
                lb = minX[2] + p*np.round(bounds[2]/parts)
                up = lb + np.round(bounds[2]/parts)
                inds = (aoX[:,2] >= lb)*(aoX[:,2] < up)
                ax2oX.append(aoX[inds])
                ax2onuX.append(aonuX[inds])
            lb = minX[2] + (parts-1)*np.round(bounds[2]/parts)
            inds = (aoX[:,2] >= lb)
            ax2oX.append(aoX[inds])
            ax2onuX.append(aonuX[inds])
    else:
        ax2oX = ax1oX
        ax2onuX = ax1onuX
    basename = particleFile.replace('npz','_')
    
    for i in range(len(ax2oX)):
        np.savez(basename + str(i) + '.npz',X=ax2oX[i],nu_X=ax2onuX[i])
        print("number of particles in " + str(i) + " is " + str(ax2oX[i].shape[0]))
        print("number of labels in " + str(i) + " is " + str(len(np.unique(ax2onuX[i]))))
    
    return 

def splitParticlesList(fileList, parts, ax0=True, ax1=True, ax2=True):
    for f in fileList:
        splitParticles(f,parts,ax0=ax0,ax1=ax1,ax2=ax2)
    return

def getFiles(filePath,suff):
    print("input into glob")
    print(filePath + '*' + suff)
    f = glob.glob(filePath + '*' + suff)
    print("number of files found is " + str(len(f)))
    return f 

def rescale(fileList, s=1e-3):
    for f in fileList:
        info = np.load(f)
        if ('X' in info.files):
            X = info['X']
            X = s*X
            np.savez(f,X=X,nu_X=info['nu_X'])
        elif ('Z' in info.files):
            Z = info['Z']
            Z = s*Z
            np.savez(f,Z=Z,nu_Z=info['nu_Z'])
        else:
            print("Z and X not in file")
    return

### KATIE: FINISH #####
def resampleWholeImage(imgFile,outpath,res=0.01,sig=0.1,C=1):
    # load image 
    im = nib.load(imgFile)
    imageO = np.asanyarray(im.dataobj).astype('float32')
    
    x0 = np.arange(imageO.shape[0])*res
    x1 = np.arange(imageO.shape[1])*res
    x2 = np.arange(imageO.shape[2])*res
    
    x0 -= np.mean(x0)
    x1 -= np.mean(x1)
    x2 -= np.mean(x2)
    
    uniqueVals = np.unique(imageO)
    numUniqueMinus0 = len(uniqueVals)-1

    return


def make_loss(X, nu_X, sig = 0.1):
    dtype = torch.cuda.FloatTensor
    tx = torch.tensor(X).type(dtype).contiguous()
    LX_i= Vi(tx)
    LX_j= Vj(tx)
    
    tnu_X = torch.tensor(nu_X).type(dtype).contiguous()
    Lnu_X_i= Vi(tnu_X)
    Lnu_X_j= Vj(tnu_X)

    D_ij = ((LX_i - LX_j)**2/sig**2).sum(dim=2)  
    K_ij = (- D_ij).exp()    
    P_ij = (Lnu_X_i*Lnu_X_j).sum(dim=2)
    c = (K_ij*P_ij).sum(dim=1).sum()

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

        E = (KZZ_ij*PZZ_ij).sum(dim=1)-2*(KZX_ij*PZX_ij).sum(dim=1)
        L = E.sum() + c
        return L
    return loss

def getWholeLoss(X,nu_X,Z,nu_Z,sig=0.1):
    dtype = torch.cuda.FloatTensor
    LX_i= Vi(torch.tensor(X).type(dtype))
    LX_j= Vj(torch.tensor(X).type(dtype))
    Lnu_X_i= Vi(torch.tensor(nu_X).type(dtype))
    Lnu_X_j= Vj(torch.tensor(nu_X).type(dtype))

    D_ij = ((LX_i - LX_j)**2/sig**2).sum(dim=2)  
    K_ij = (- D_ij).exp()    
    P_ij = (Lnu_X_i*Lnu_X_j).sum(dim=2)

    LZ_i = Vi(torch.tensor(Z).type(dtype))
    LZ_j = Vj(torch.tensor(Z).type(dtype))
    Lnu_Z_i = Vi(torch.tensor(nu_Z).type(dtype))
    Lnu_Z_j = Vj(torch.tensor(nu_Z).type(dtype))
    
    DZZ_ij = ((LZ_i - LZ_j)**2/sig**2).sum(dim=2)  
    KZZ_ij = (- DZZ_ij).exp()    
    PZZ_ij = (Lnu_Z_i*Lnu_Z_j).sum(dim=2)

    DZX_ij = ((LZ_i - LX_j)**2/sig**2).sum(dim=2) 
    KZX_ij = (- DZX_ij).exp() 
    PZX_ij = (Lnu_Z_i*Lnu_X_j).sum(dim=2)
    
    E = (KZZ_ij*PZZ_ij).sum(dim=1)-2*(KZX_ij*PZX_ij).sum(dim=1) + (K_ij*P_ij).sum(dim=1)
    L = E.sum()
    return L
  
#########################################
def estimateSubSampleImg(img,outpath,ax=2,indS=920,indF=940,res=0.01,sig=0.1,C=1):
    #sig = 0.1
    #C = 1
    dtype = torch.cuda.FloatTensor
    lossTrack = []
       
    X,nu_X,vol = make_target3DImage(img,ax=2,indS=indS,indF=indF,res=res)
    N = round(C*vol/(sig**3))
    print("number of points in subsample is " + str(N))
    
    tokeep = np.sum(nu_X,axis=-1) > 0 
    print("total points before")
    print(X.shape[0])
    X = X[tokeep,...]
    nu_X = nu_X[tokeep,...]
    print("total points now")
    print(X.shape[0])
    maxIndnu = np.argmax(nu_X,axis=-1)+1
    vtf.writeVTK(X,[maxIndnu],['MAX_VAL_NU'],outpath+'_originalXnu_X.vtk',polyData=None)

    Z,nu_Z = make_subsample(X,nu_X, N)
    loss = make_loss(X,nu_X)

    LX_i= Vi(torch.tensor(X).type(dtype))
    LX_j= Vj(torch.tensor(X).type(dtype))
    Lnu_X_i= Vi(torch.tensor(nu_X).type(dtype))
    Lnu_X_j= Vj(torch.tensor(nu_X).type(dtype))

    D_ij = ((LX_i - LX_j)**2/sig**2).sum(dim=2)  
    K_ij = (- D_ij).exp()    
    P_ij = (Lnu_X_i*Lnu_X_j).sum(dim=2)
    
    Zal_Z = np.hstack((Z,np.sqrt(nu_Z)))
    vtf.writeVTK(Z,[np.argmax(nu_Z,axis=-1)+1,np.sum(nu_Z,axis=-1)],['MAX_VAL_NU','TOTAL_MASS'],outpath+'_resampleZnu_ZwC' + str(C) + '_sig' + str(sig) + '.vtk',polyData=None)
    
    def make_loss2(X, n_X, Z):
        LX_i= Vi(torch.tensor(X).type(dtype).contiguous())
        LX_j= Vj(torch.tensor(X).type(dtype).contiguous())
        Lnu_X_i= Vi(torch.tensor(nu_X).type(dtype).contiguous())
        Lnu_X_j= Vj(torch.tensor(nu_X).type(dtype).contiguous())

        D_ij = ((LX_i - LX_j)**2/sig**2).sum(dim=2)  
        K_ij = (- D_ij).exp()    
        P_ij = (Lnu_X_i*Lnu_X_j).sum(dim=2)
        c= (K_ij*P_ij).sum(dim=1).sum()

        tz = torch.tensor(Z).type(dtype).contiguous()
        LZ_i, LZ_j= Vi(tz), Vj(tz)
        DZZ_ij = ((LZ_i - LZ_j)**2/sig**2).sum(dim=2)  
        KZZ_ij = (- DZZ_ij).exp()    

        print('c=',c)

        def loss(tal_Z):   
            Lnu_Z_i, Lnu_Z_j = Vi(tal_Z**2), Vj(tal_Z**2)

            PZZ_ij = (Lnu_Z_i*Lnu_Z_j).sum(dim=2)

            DZX_ij = ((LZ_i - LX_j)**2/sig**2).sum(dim=2) 
            KZX_ij = (- DZX_ij).exp() 
            PZX_ij = (Lnu_Z_i*Lnu_X_j).sum(dim=2)
      
            E = (KZZ_ij*PZZ_ij).sum(dim=1)-2*(KZX_ij*PZX_ij).sum(dim=1)
            L = E.sum() +c
            return L
        return loss

    
    def optimize(Z, nZ, nb_iter = 20, flag = 'all'):
        if flag == 'all':
            loss = make_loss(X,nu_X)
            Zal_Z = np.hstack((Z,np.sqrt(nu_Z)))
            p0 = torch.tensor(Zal_Z).type(dtype).contiguous().requires_grad_(True)
        else:
            loss = make_loss2(X,nu_X, Z)
            al_Z = np.sqrt(nu_Z)
            p0 = torch.tensor(al_Z).type(dtype).contiguous().requires_grad_(True)

        print('p0', p0.is_contiguous())
        optimizer = torch.optim.LBFGS([p0], max_eval=20, max_iter=40, line_search_fn = 'strong_wolfe')
    
        def closure():
            optimizer.zero_grad()
            L = loss(p0)
            print("loss", L.detach().cpu().numpy())
            L.backward()
            return L

        for i in range(nb_iter):
            print("it ", i, ": ", end="")
            optimizer.step(closure)

        if flag == 'all':
            nZ = p0[:,0:3].detach().cpu().numpy()
            nnu_Z = p0[:,3::].detach().cpu().numpy()**2
        else:
            nZ = Z
            nnu_Z = p0.detach().cpu().numpy()**2
        return nZ, nnu_Z
    '''
    p0 = torch.tensor(Zal_Z).type(dtype).requires_grad_(True)
    optimizer = torch.optim.LBFGS([p0],max_eval=20,max_iter=40, line_search_fn = 'strong_wolfe')
    
    def closure():
        optimizer.zero_grad()
        L = loss(p0)
        print("loss", L.detach().cpu().numpy())
        lossTrack.append(L.detach().cpu().numpy())
        L.backward()
        return L

    for i in range(20):
        print("it", i, ":", end="")
        optimizer.step(closure)
    
    nZ = p0[:,0:3].detach().cpu().numpy()
    nnu_Z = (p0[:,3::].detach().cpu().numpy())**2
    '''
    
    Z, nu_Z = optimize(Z, nu_Z, nb_iter = 3, flag = '')
    nZ, nnu_Z = optimize(Z, nu_Z, nb_iter = 40, flag = 'all')

    maxInd = np.argmax(nnu_Z,axis=-1)+1
    
    vtf.writeVTK(nZ,[maxInd,np.sum(nnu_Z,axis=-1)],['MAX_VAL_NU','TOTAL_MASS'],outpath+'_optimalZnu_ZwC' + str(C) + '_sig' + str(sig) + '.vtk',polyData=None)
    #ll = getWholeLoss(X,nu_X,nZ,nnu_Z)
                        
    #print("final loss is " + str(ll.detach().cpu().numpy()))
    f,ax = plt.subplots()
    ax.plot(np.arange(len(lossTrack)),np.asarray(lossTrack))
    ax.set_xlabel('iterations')
    ax.set_ylabel('cost')
    f.savefig(outpath+'_optimalZcost_sig' + str(sig) + '_C' + str(C) + '.png',dpi=300)
    
    return

def project3DLoop(X,nu_X,eps, sigma, nb_iter0, nb_iter1,outpath,Z=None,nu_Z=None):
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
        return [rangesXX_ij, rangesZZ_ij, rangesZX_ij], [X_labels, Z_labels]

    def make_loss(tX, tnu_X, len_Z, dim_nu_Z, ranges):
        rangesXX_ij, rangesZZ_ij, rangesZX_ij = ranges # Getting the ranges
        LX_i = LazyTensor(tX[:,None,:])
        LX_j = LazyTensor(tX[None,:,:])

        Lnu_X_i = LazyTensor(tnu_X[:,None,:])
        Lnu_X_j = LazyTensor(tnu_X[None,:,:])

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

        Lnu_X_i = LazyTensor(tnu_X[:,None,:])
        Lnu_X_j = LazyTensor(tnu_X[None,:,:])

        D_ij = ((LX_i - LX_j)**2/sig**2).sum(dim=2)  
        K_ij = (- D_ij).exp()
        P_ij = (Lnu_X_i*Lnu_X_j).sum(dim=2)
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
    tnu_X = torch.tensor(nu_X).type(dtype)
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
    print("sum tnu_X", tnu_X.sum())
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

    # Display of the result
    maxInd = np.argmax(nnu_Z,axis=-1)+1
    vtf.writeVTK(nZ,[maxInd,np.sum(nnu_Z,axis=-1)],['MAX_VAL_NU','TOTAL_MASS'],outpath+'_optimalZnu_ZwC' + str(C) + '_sig' + str(sig) + 'eps' + str(eps) + '.vtk',polyData=None)
    np.savez(outpath+'_optimalZnu_ZwC' + str(C) + '_sig' + str(sig) + '.npz',Z=nZ, nu_Z=nnu_Z)
                            
    f,ax = plt.subplots()
    ax.plot(np.arange(len(lossTrack)),np.asarray(lossTrack))
    ax.set_xlabel('iterations')
    ax.set_ylabel('cost')
    f.savefig(outpath+ '_optimalZcost_sig' + str(sig) + '_C' + str(C) + 'eps' + str(eps) +'.png',dpi=300)
    
    return nZ, nnu_Z