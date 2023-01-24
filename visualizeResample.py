# Libraries to Import
import numpy as np
import math
import sys
import matplotlib.pyplot as plt
import os
import glob
import scipy.interpolate as spi
import scipy as sp
# these methods from scipy will be used for displaying some images
from scipy.linalg import eigh
from scipy.stats import norm

# Tools for Generating Paths
from numpy import linalg as LA
from itertools import combinations
from scipy.special import comb
from scipy import stats

# Tools for caching
from os import path

# Tools for saving images
import scipy.misc
from scipy import io

# Tools for Reading in File
import nibabel as nib
from nibabel import processing
from nibabel import funcs

import sys
sys.path.append('/cis/home/kstouff4/Documents/SurfaceTools/')

import deformSegmentations3D as ds
import vtkFunctions as vtf

import random

import ntpath
from sys import path as sys_path
sys_path.append('/cis/home/kstouff4/Documents/MeshRegistration/master/py-lddmm/')
sys_path.append('/cis/home/kstouff4/Documents/MeshRegistration/master/py-lddmm/base')
import os
from base import loggingUtils
from base import vtk_fields
from vtk_fields import vtkFields
import multiprocessing as mp
from multiprocessing import Pool
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import imageio.v3 as iio
import cv2
from skimage.segmentation import watershed
from base.meshes import buildImageFromFullListHR, buildMeshFromCentersCounts, buildMeshFromImageData, buildMeshFromCentersCountsMinMax
from PIL import Image
Image.MAX_IMAGE_PIXELS
Image.MAX_IMAGE_PIXELS=1e10 # forget attack
import nibabel as nib
import re

import torch
from pykeops.torch import LazyTensor

from pykeops.torch.cluster import sort_clusters
from pykeops.torch.cluster import cluster_ranges_centroids
from pykeops.torch.cluster import grid_cluster
from pykeops.torch.cluster import from_matrix
import pykeops

try:
    from vtk import vtkCellArray, vtkPoints, vtkPolyData, vtkVersion,\
        vtkLinearSubdivisionFilter, vtkQuadricDecimation,\
        vtkWindowedSincPolyDataFilter, vtkImageData, VTK_FLOAT,\
        vtkDoubleArray, vtkContourFilter, vtkPolyDataConnectivityFilter,\
        vtkCleanPolyData, vtkPolyDataReader, vtkUnstructuredGridReader, vtkOBJReader, vtkSTLReader,\
        vtkDecimatePro, VTK_UNSIGNED_CHAR, vtkPolyDataToImageStencil,\
        vtkImageStencil
    from vtk.util.numpy_support import vtk_to_numpy
    gotVTK = True
except ImportError:
    v2n = None
    print('could not import VTK functions')
    gotVTK = False

##################################################################################################
def groupXbyLabel(oneHotEncode,X):
    d = oneHotEncode.shape[-1]
    listOfX = []
    for i in range(d):
        listOfX.append(X[oneHotEncode[:,i]])
    return listOfX

def oneHot(nu_X,tot=673):
    '''
    Make nu_X into full nu_X by expanding single dimension to maximum number
    Make nu_Z into subsampled nu_Z 
    '''
    nnu_X = np.zeros((nu_X.shape[0],tot)).astype('bool_') # assume nu_Z has full spectrum
    nnu_X[np.arange(nu_X.shape[0]),np.squeeze(nu_X-1).astype(int)] = 1
    print(np.unique(nu_X))
    
    nonZeros = np.sum(nnu_X,axis=0)
    indsToKeep = np.where(nonZeros > 0)
    print("total is " + str(len(indsToKeep[0]))) # 0 based with maximum = 1 less than dimension
    print(indsToKeep[0])
    
    nnu_X = nnu_X[:,indsToKeep[0]]

    return nnu_X, indsToKeep[0]

def getDensity(npzTrans,centerName='X',featureName='nu_X',sigma=0.2):
    dtype = torch.cuda.FloatTensor
    pykeops.clean_pykeops()
    p = np.load(npzTrans)
    X = p[centerName]
    nu_Xo = p[featureName]
    nu_X, indsToKeep = oneHot(nu_Xo)
    Xlist = groupXbyLabel(nu_X,X) # returns list of X centers each with single label 
    #nu_X = np.zeros((X.shape[0],int(np.max(nu_Xo)))).astype('bool_')
    #nu_X[np.arange(nu_X.shape[0]),np.squeeze(nu_Xo-1).astype(int)] = 1
    print(nu_Xo.shape)
    print(X.shape)
    
    xMin = np.floor(np.min(X,axis=0))
    xMax = np.ceil(np.max(X,axis=0))
    
    nSX = np.round((xMax[0] - xMin[0])/sigma)
    nSY = np.round((xMax[1] - xMin[1])/sigma)
    nSZ = np.round((xMax[2] - xMin[2])/sigma)
    
    def dens_est(x,yList,ranges,sig=sigma,weightsList=None):
        '''
        y should be where original points were 
        '''
        if weightsList is None:
            wTot = 0
            wList = []
            for yy in yList:
                wList.append(np.ones((yy.shape[0],1)))
                wTot += yy.shape[0]
        else:
            wList = weightsList
            wTot = 0
            for w in wList:
                wTot += np.sum(w) ## NOT IMPLEMENTED YET (ASSUME ALL SAME WEIGHTS; WILL NEED TO FACTOR THIS INTO DENSITY)
        LX_i = LazyTensor(x[:,None,:])
        rhoList = []
        for i in range(len(yList)):
            y = yList[i]
            w = np.squeeze(wList[i])[...,None] # ensure is points x 1 
            tW = torch.tensor(w).type(dtype)
            LX_j = LazyTensor(y[None,:,:])
            Lw_j = LazyTensor(tW[None,:,:])
        #Lnu_y = LazyTensor(nu_y[None,...].type(dtype)) # 1 x Feat x Full # memory errors 

            D_ij = ((LX_i - LX_j)**2/sig**2).sum(dim=2)  # 1 x S x Full
            K_ij = (- D_ij).exp() # 1 x S x Full
            K_ij = Lw_j*K_ij # w * pi
            K_ij.ranges = ranges[i]
            rho = K_ij.sum(dim=1) # sum over i = w_j
            rhoList.append(rho)
            '''
            f = wTot/np.sum(rho)
            rho = rho*f # conservation of mass # w_j : total mass is conserved 
            print(rho.shape)
            print(K_ij.shape)
            K_ij*f # denotes sum for each x (and 
            inds = K_ij.argmax(dim=1)
            feats = nu_y[inds.cpu().numpy(),:]
            '''
        # K_ij = K_ij*Lnu_y  # memory errors
        #K_ij.ranges = ranges
        #feats = K_ij.sum(dim=1)
        # normalize 
        nu_Z = np.zeros((x.shape[0],len(yList)))
        for i in range(len(rhoList)):
            nu_Z[:,i] = np.squeeze(rhoList[i].cpu().numpy())
        totalRho = np.sum(nu_Z,axis=-1)
        f = wTot/np.sum(totalRho)
        totalRho = totalRho*f
        totalRhore = np.reciprocal(totalRho, where=totalRho!=0)
        nu_Z = nu_Z*f*totalRhore[...,None]
        
        return np.squeeze(totalRho), np.squeeze(nu_Z)  # (S x Features)
    
    def make_ranges(Xlist,Z,eps=3*sigma, sig=sigma):
      # Here X and Z are torch tensors
        rangesXZlist = []
        X_labelslist = []
        Z_labels = grid_cluster(Z, eps) 
        Z_ranges, Z_centroids, _ = cluster_ranges_centroids(Z, Z_labels)
        for X in Xlist:
            X_labels = grid_cluster(X, eps) 
            X_ranges, X_centroids, _ = cluster_ranges_centroids(X, X_labels)
            print(X_centroids.shape)
            a = np.sqrt(3)
            D = ((Z_centroids[:, None, :] - X_centroids[None, :, :]) ** 2).sum(dim=2)
            keep = D <(a*eps+4* sig) ** 2
            rangesZX_ij = from_matrix(Z_ranges, X_ranges, keep)
            areas = (Z_ranges[:, 1] - Z_ranges[:, 0])[:, None] * (X_ranges[:, 1] 
                            - X_ranges[:, 0])[None, :]
            '''
            D = ((X_centroids[:, None, :] - Z_centroids[None, :, :]) ** 2).sum(dim=2)
            keep = D <(a*eps+4* sig) ** 2
            rangesXZ_ij = from_matrix(X_ranges, Z_ranges, keep)
            areas = (X_ranges[:, 1] - X_ranges[:, 0])[:, None] * (Z_ranges[:, 1] 
                        - Z_ranges[:, 0])[None, :]
            '''
            total_area = areas.sum()  # should be equal to N*M
            sparse_area = areas[keep].sum()
            print(
            "We keep {:.2e}/{:.2e} = {:2d}% of the original kernel matrix.".format(
              sparse_area, total_area, int(100 * sparse_area / total_area)))
            print("")
            rangesXZlist.append(rangesZX_ij)
            X_labelslist.append(X_labels)
        return rangesXZlist, X_labelslist, Z_labels
    
    for mu in [2,1,0.5]:
        # create sampling grid based on sigma resolution 
        sX = torch.linspace(xMin[0],xMax[0],int(mu*nSX)).type(dtype)
        sY = torch.linspace(xMin[1],xMax[1],int(mu*nSY)).type(dtype)
        sZ = torch.linspace(xMin[2],xMax[2],int(mu*nSZ)).type(dtype)
        
        SX,SY,SZ = torch.meshgrid(sX,sY,sZ)
        t = torch.stack((SX.contiguous().view(-1), SY.contiguous().view(-1),SZ.contiguous().view(-1)),dim=1)
    
        tXlist = []
        for X in Xlist:
            tXlist.append(torch.tensor(X).type(dtype))
                          
        #tC = torch.tensor(X).type(dtype)
        #tnuC = torch.tensor(nu_X).type(torch.bool)
        print("Making ranges")
        rangesList, X_labelsList, C_labels = make_ranges(tXlist,t, eps=3*sigma, sig=sigma)
        print("Sorting X and nu_X")
        tT, _ = sort_clusters(t,C_labels)
        for i in range(len(tXlist)):
            tXlist[i], _ = sort_clusters(tXlist[i], X_labelsList[i]) # sorting the labels

        rho, feats = dens_est(tT,tXlist,rangesList,sig=sigma)
        print(rho.shape)
        #rho = rhoC.cpu().numpy()
        points = t.cpu().numpy()
        n = npzTrans.replace('.npz','gauss_sigma' + str(sigma) + '_fact' + str(mu) + '.npz')
        np.savez(n,rho=rho,grid=points,feats=feats)
    
    
        print("min and max")
        print(str(np.min(rho)) + ", " + str(np.max(rho)))
        
        maxFeat = np.argmax(feats,axis=-1)
        featList = [rho[rho > 0]]
        featList.append(maxFeat[rho > 0])
        nameList = ['LOCAL_DENSITY']
        nameList.append('MAX_REGION')
       
        for f in range(feats.shape[-1]):
            featList.append(feats[rho > 0,f])
            nameList.append('LABEL_ORIG_' + str(indsToKeep[f]))
            
        vtf.writeVarifoldVTK(points[rho > 0],featList,nameList,n.replace('.npz','.vtk'))

    return

def writeParticleVTK(npz,varName='Z'):
    '''
    Assume npz has form of features (nu_Z or nu_X) and form of locations (Z or X) which is given in varName
    '''
    info = np.load(npz)
    X = info[varName]
    nuX = info['nu_' + varName]
    print("nuX shape")
    print(nuX.shape)
    
    # write scalars
    #### total mass (sum of the nuX)
    #### maxVal of total (ind of maxVal)
    #### maxVal of nonzero (remapped to contiguous set)
    # write fields
    #### value for each of the total feature dimensions 
    
    maxTot = np.argmax(nuX,axis=-1)
    totMass = np.sum(nuX,axis=-1)
    s = np.sum(nuX,axis=0)
    indsToKeep = s > 0
    print(indsToKeep.shape)
    nuXsub = nuX[:,indsToKeep]
    print(nuXsub.shape)
    print("keeping " + str(np.sum(indsToKeep)))
    maxSub = np.argmax(nuXsub,axis=-1)
    
    fileName = npz.replace('.npz','_full.vtk')
    with open(fileName, 'w') as fvtkout:
        fvtkout.write('# vtk DataFile Version 3.0\nMesh Data\nASCII\nDATASET UNSTRUCTURED_GRID\n')
        fvtkout.write('\nPOINTS {0: d} float'.format(X.shape[0]))
        for ll in range(X.shape[0]):
            fvtkout.write('\n')
            for kk in range(X.shape[-1]):
                fvtkout.write(f'{X[ll,kk]: f} ')
            if X.shape[-1] == 2:
                fvtkout.write('0')

        v = vtkFields('POINT_DATA', X.shape[0], scalars = {'alpha':totMass,'maxValTot':maxTot,'maxValNonZero':maxSub}, fields = {'nu_' + varName:nuX})
        v.write(fvtkout)
    return


    
