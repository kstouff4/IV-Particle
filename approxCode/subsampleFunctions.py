import numpy as np
from sys import path as sys_path
sys_path.append('/cis/home/kstouff4/Documents/SurfaceTools/')
import vtkFunctions as vtf
import pandas as pd

####################################################################
# File Description

'''
Contains functions for generating starting approximation for Varifold Approximation

Conventions:
X, nu_X \in R^d x R^f is assumed to be at the finest scale and is either:
    - semi-discrete (\sum_i \delta(x_i) \otimes \nu_i) where \nu_i is a measure over R^f (e.g. where measures are localized to cell centers and feature space indicates copies of different genes)
    - discrete (\sum_i \delta_(x_i) \otimes \alpha_i\delta_l) where \alpha_i indicates the mass that is prescribed only to one feature (e.g. gene type) in the feature space 
    
Z, nu_Z \in R^d x R^f is assumed to be at a coarser scale and stored always as:
    -semi-discrete (\sum_i \delta(x_i) \otimes \nu_i) where \nu_i is a measure over R^f

For optimal memory usage, nu_Z is reduced to a subset of R^f when working on subsets (e.g. slices, slabs) of an entire dataset
'''
####################################################################

def oneHot(nu_X,nu_Z,zeroBased=False):
    '''
    Make nu_X into full nu_X by expanding single dimension to maximum number
    Make nu_Z into subsampled nu_Z 
    '''
    nnu_X = np.zeros((nu_X.shape[0],nu_Z.shape[-1])).astype('bool_') # assume nu_Z has full spectrum
    if (zeroBased):
        nnu_X[np.arange(nu_X.shape[0]),np.squeeze(nu_X).astype(int)] = 1
    else:
        nnu_X[np.arange(nu_X.shape[0]),np.squeeze(nu_X-1).astype(int)] = 1
    print(np.unique(nu_X))
    
    nonZeros = np.sum(nnu_X,axis=0)+np.sum(nu_Z,axis=0)
    indsToKeep = np.where(nonZeros > 0)
    print("total is " + str(len(indsToKeep[0]))) # 0 based with maximum = 1 less than dimension
    print(indsToKeep[0])
    
    nnu_X = nnu_X[:,indsToKeep[0]]
    nnu_Z = nu_Z[:,indsToKeep[0]]

    return nnu_X,nnu_Z,indsToKeep[0]

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

def makeOneHot(nu,maxVal=673,zeroBased=False):
    nu1 = np.zeros((nu.shape[0],maxVal)).astype('float32')
    print("min should not be 0: " + str(np.min(nu)))
    if (zeroBased):
        nu1[np.arange(nu.shape[0]),np.squeeze(nu).astype(int)] = 1
    else:
        nu1[np.arange(nu.shape[0]),np.squeeze(nu-1).astype(int)] = 1
    return nu1

def reverseOneHot(nu_Z,indsToKeep,maxVal=673):
    '''
    assume IndsToKeep indicate the mapping of original labels
    '''
    nnuZ = np.zeros((nu_Z.shape[0],maxVal))
    nnuZ[:,indsToKeep] = nu_Z
    return nnuZ

def makeSubsampleFun(X, nu_X, sig, C=1.2,dis=True, maxV=673):
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
    if (dis):
        nu_Z = makeOneHot(nu_Z,maxVal=maxV)*X.shape[0]/N
    return Z, nu_Z

def addOverhead(Xfile,Zfile,overhead=0.1,maxV=673):
    zinfo = np.load(Zfile)
    Z = zinfo['Z']
    nu_Z = zinfo['nu_Z']
    xinfo = np.load(Xfile)
    nu_X = xinfo['X']
    nnu_X,nnu_Z,indsToKeep = oneHot(nu_X,nu_Z) # return only those labels with > 0 mass
    nu_Z = nnu_Z + overhead
    nu_Z = reverseOneHot(nu_Z,indsToKeep,maxV)
    np.savez(Zfile.replace('.npz','_plus' + str(overhead) + '.npz'),Z=Z,nu_Z=nu_Z)
    return

def makeSubsample(Xfile,sig,savename,xtype='discrete',ztype='semi-discrete',overhead=0.1,maxV=673,C=1.2):
    '''
    Types of sampling:
        - xtype = semi-discrete, then z type = semi-discrete
        - xtype = discrete and z type = discrete
        - xtype = discrete and z type = semi-discrete (e.g. add overhead )
    '''
    
    xinfo = np.load(Xfile)
    X=xinfo['X']
    nu_X = xinfo['nu_X']
    if (xtype == 'discrete'):
        Z, nu_Z = makeSubsampleFun(X,nu_X,sig,C=C,dis=True,maxV=maxV)
        # write original Z 
        maxInd = np.argmax(nu_Z,axis=-1)+1
        vtf.writeVTK(Z,[maxInd,np.sum(nu_Z,axis=-1)],['MAX_VAL_NU','TOTAL_MASS'],savename+'_originalZnu_ZwC' + str(C) + '_sig' + str(sig) + '_discrete.vtk',polyData=None)
        np.savez(savename+'_originalZnu_ZwC' + str(C) + '_sig' + str(sig) + '_discrete.npz',Z=Z, nu_Z=nu_Z)
        if (ztype == 'semi-discrete'):
            nnu_Z,indsToKeep = oneHotMemorySave(nu_X,nu_Z)
            nu_Z = nnu_Z + overhead
            nu_Z = reverseOneHot(nu_Z,indsToKeep,maxV)
            np.savez(savename+'_originalZnu_ZwC' + str(C) + '_sig' + str(sig) + '_semidiscrete_plus' + str(overhead) + '.npz',Z=Z, nu_Z=nu_Z)
            maxInd = np.argmax(nu_Z,axis=-1)+1
            vtf.writeVTK(Z,[maxInd,np.sum(nu_Z,axis=-1)],['MAX_VAL_NU','TOTAL_MASS'],savename+'_originalZnu_ZwC' + str(C) + '_sig' + str(sig) + '_discrete.vtk',polyData=None)
    elif (xtype == 'semi-discrete'):
        Z, nu_Z = makeSubsampleFun(X,nu_X,sig,C=C,dis=False,maxV=maxV)
        # write original Z 
        maxInd = np.argmax(nu_Z,axis=-1)+1
        vtf.writeVTK(Z,[maxInd,np.sum(nu_Z,axis=-1)],['MAX_VAL_NU','TOTAL_MASS'],savename+'_originalZnu_ZwC' + str(C) + '_sig' + str(sig) + '_semidiscrete.vtk',polyData=None)
        np.savez(savename+'_originalZnu_ZwC' + str(C) + '_sig' + str(sig) + '_semidiscrete.npz',Z=Z, nu_Z=nu_Z)
    
    return

def makeSubsampleStratified(Xfile,coordName,nuName,sig,savename,xtype='discrete',ztype='semi-discrete',overhead=0.1,maxV=702,C=1.2,dim=2,z=0,saveX=True):
    '''
    Sampling of same types as in makeSubsample, but with weighting geographically in subset
    
    Algorithm:
        - Divide bounding box into squares (cubes) of area / volume sigma^dim
        - Select np.round(C*np.log(counts + 1)) particles from each square (cube) based on counts in the cube 
        
    coordName = name of variable storing coordinates (assume N x 2/3)
    nuName = name of variable storing feature value (assume N x 1)
    
    '''
    if (coordName == 'geneInd'):
        zeroBased = True
        print("features are zero-based")
    else:
        zeroBased = False
    info = np.load(Xfile)
    print(info.files)
    coords = info[coordName]
    print("shape of coords")
    if coords.shape[-1] > dim:
        coords = coords[:,0:dim]
    
    print(coords.shape)
    nuX = np.squeeze(info[nuName])
    if (saveX):
        X = coords
        if dim == 2:
            xx = np.zeros((coords.shape[0],3))
            xx[:,0:2] = coords
            xx[:,-1] = z
            X = xx
        np.savez(savename+'_XnuX.npz',X=X,nu_X=nuX)
    
    # random shuffle to do random sample
    if (dim == 2):
        allInfo = np.stack((coords[:,0],coords[:,1],nuX),axis=-1)
        np.random.shuffle(allInfo)
        coords = allInfo[:,0:2]
        nuX = allInfo[:,-1]
    elif (dim == 3):
        allInfo = np.stack((coords[:,0],coords[:,1],coords[:,2]),axis=-1)
        np.random.shuffle(allInfo)
        coords = allInfo[:,0:3]
        nuX = allInfo[:,-1]
    
    # divide into cubes of size sig
    if (dim == 2):
        coords_labels = np.floor((coords - np.floor(np.min(coords,axis=0)))/sig).astype(int) # minimum number of cubes in x and y 
        totCubes = (np.max(coords_labels[:,0])+1)*(np.max(coords_labels[:,1])+1)
    elif (dim == 3):
        coords_labels = np.floor((coords - np.floor(np.min(coords,axis=0)))/sig).astype(int) # minimum number of cubes in x and y 
        totCubes = (np.max(coords_labels[:,0])+1)*(np.max(coords_labels[:,1])+1) + (np.max(coords_labels[:,2]) + 1)
    
    # coords_labels_tot gives the index of the cube for each of original measurements
    '''
    # Pandas version
    df = pd.DataFrame()
    df['x'] = coords[:,0]
    df['y'] = coords[:,1]
    df['nu'] = nuX
    df['cX'] = coords_labels[:,0]
    df['cY'] = coords_labels[:,1]
    if (dim == 3):
        df['z'] = coords[:,2]
        df['cZ'] = coords_labels[:,2]
        dff = df.groupby(['cX','cY','cZ']).sample(n=1) # select 1 from each group (could do frac=0.1 to get 10% total in each group)
        Z = dff['x','y','z'].to_numpy()
        nuZ = dff['nu'].to_numpy()
    elif (dim == 2):
        dff = df.groupby(['cX','cY']).sample(n=1)
        Z = dff['x','y'].to_numpy()
        nuZ = dff['nu'].to_numpy()
    '''
    
    # Numpy version 
    uInds,sample = np.unique(coords_labels,return_index=True,axis=0) # returns first occurrence of each cube 
    nuZ = nuX[sample]
    Z = coords[sample,...]
    if dim == 2:
        zz = np.zeros((Z.shape[0],3))
        zz[:,0:2] = Z
        zz[:,-1] = z
        Z = zz
    
    nu_Z = makeOneHot(nuZ,maxVal=maxV,zeroBased=zeroBased)*coords.shape[0]/nuZ.shape[0] # weigh points with appropriate mass and make large array
    
    maxInd = np.argmax(nu_Z,axis=-1)+1
    vtf.writeVTK(Z,[maxInd,np.sum(nu_Z,axis=-1)],['MAX_VAL_NU','TOTAL_MASS'],savename+'_originalZnu_ZwC' + str(C) + '_sig' + str(sig) + '_discrete.vtk',polyData=None)
    np.savez(savename+'_originalZnu_ZwC' + str(C) + '_sig' + str(sig) + '_discrete.npz',Z=Z, nu_Z=nu_Z)
    if (ztype == 'semi-discrete'):
        nnu_Z,indsToKeep = oneHotMemorySave(nuX,nu_Z,zeroBased=zeroBased)
        nu_Z = nnu_Z + overhead
        nu_Z = reverseOneHot(nu_Z,indsToKeep,maxV)
        np.savez(savename+'_originalZnu_ZwC' + str(C) + '_sig' + str(sig) + '_semidiscrete_plus' + str(overhead) + '.npz',Z=Z, nu_Z=nu_Z)
        maxInd = np.argmax(nu_Z,axis=-1)+1
        vtf.writeVTK(Z,[maxInd,np.sum(nu_Z,axis=-1)],['MAX_VAL_NU','TOTAL_MASS'],savename+'_originalZnu_ZwC' + str(C) + '_sig' + str(sig) + '_semidiscrete_plus' + str(overhead) + '.vtk',polyData=None)
    
    return


        


        


    

