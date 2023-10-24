import numpy as np
from sys import path as sys_path
sys_path.append('/cis/home/kstouff4/Documents/SurfaceTools/')
import vtkFunctions as vtf
import pandas as pd

sys_path.append('../varap/io/')
from writeOut import writeParticleVTK

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
    nu_Z = nu_X[sub_ind,...] # start weights off as fraction of what you started with
    if (dis):
        nu_Z = makeOneHot(nu_Z,maxVal=maxV)*X.shape[0]/N
    else:
        # weigh nu_X with the total delta of mass missing
        xMass = np.sum(nu_X)
        zMass = np.sum(nu_Z)
        c = xMass/zMass
        nu_Z = c*nu_Z
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
    X=xinfo[xinfo.files[0]]
    nu_X = xinfo[xinfo.files[1]]
    print(X.shape)
    print(nu_X.shape)
    if (xtype == 'discrete'):
        Z, nu_Z = makeSubsampleFun(X,nu_X,sig,C=C,dis=True,maxV=maxV)
        # write original Z 
        maxInd = np.argmax(nu_Z,axis=-1)+1
        vtf.writeVTK(Z,[maxInd,np.sum(nu_Z,axis=-1)],['MAX_VAL_NU','TOTAL_MASS'],savename+'_originalZnu_ZwC' + str(C) + '_sig' + str(sig) + '_discrete.vtk',polyData=None)
        np.savez(savename+'_originalZnu_ZwC' + str(C) + '_sig' + str(sig) + '_discrete.npz',Z=Z, nu_Z=nu_Z)
        if (ztype == 'semi-discrete'):
            nnu_Z,indsToKeep = oneHotMemorySave(nu_X,nu_Z)
            nu_Z = nnu_Z + overhead
            nu_Z = nu_Z*(np.sum(nnu_Z)/np.sum(nu_Z))
            nu_Z = reverseOneHot(nu_Z,indsToKeep,maxV)
            np.savez(savename+'_originalZnu_ZwC' + str(C) + '_sig' + str(sig) + '_semidiscrete_plus' + str(overhead) + '.npz',Z=Z, nu_Z=nu_Z)
            print("total mass before")
            print(X.shape[0])
            print("total mass after")
            print(np.sum(nu_Z))
            maxInd = np.argmax(nu_Z,axis=-1)+1
            vtf.writeVTK(Z,[maxInd,np.sum(nu_Z,axis=-1)],['MAX_VAL_NU','TOTAL_MASS'],savename+'_originalZnu_ZwC' + str(C) + '_sig' + str(sig) + 'semidiscrete.vtk',polyData=None)
    elif (xtype == 'semi-discrete'):
        Z, nu_Z = makeSubsampleFun(X,nu_X,sig,C=C,dis=False,maxV=maxV)
        nnu_Z = nu_Z + overhead
        nu_Z = nnu_Z*(np.sum(nu_Z)/np.sum(nnu_Z))
        print("sum of nuX vs nuZ")
        print(np.sum(nu_X))
        print(np.sum(nu_Z))
        # write original Z 
        maxInd = np.argmax(nu_Z,axis=-1)+1
        vtf.writeVTK(Z,[maxInd,np.sum(nu_Z,axis=-1)],['MAX_VAL_NU','TOTAL_MASS'],savename+'_originalZnu_ZwC' + str(C) + '_sig' + str(sig) + '_semidiscrete.vtk',polyData=None)
        np.savez(savename+'_originalZnu_ZwC' + str(C) + '_sig' + str(sig) + '_semidiscrete.npz',Z=Z, nu_Z=nu_Z)
    
    return savename+'_originalZnu_ZwC' + str(C) + '_sig' + str(sig) + '_semidiscrete.npz'

def makeSubsampleStratified(Xfile,coordName,nuName,sig,savename,xtype='discrete',ztype='semi-discrete',overhead=0.1,maxV=702,Co=1.2,dim=3,z=None,saveX=True,zeroBased=True,alpha=0.5):
    '''
    Sampling of same types as in makeSubsample, but with weighting geographically in subset
    
    Algorithm:
        - Divide bounding box into squares (cubes) of area / volume sigma^dim
        - Select np.round(C*np.log(counts + 1)) particles from each square (cube) based on counts in the cube 
        
    coordName = name of variable storing coordinates (assume N x 2/3)
    nuName = name of variable storing feature value (assume N x 1)
    
    ztype \in {'discrete', 'semi-discrete', 'uniform'}
    
    '''
    C = ((1.0/alpha)**3) * Co
    if (zeroBased is None):
        if (coordName == 'geneInd'):
            zeroBased = True
            print("features are zero-based")
        else:
            zeroBased = False
    info = np.load(Xfile)
    print(info.files)
    coords = info[info.files[0]]
    print("shape of coords")
    if z is None:
        z = np.mean(coords[:,-1])
    if coords.shape[-1] > dim:
        coords = coords[:,0:dim]
    
    print(coords.shape)
    nuX = np.squeeze(info[info.files[1]])
    print("nuX shape, ", nuX.shape)
    print("sum of nuX, ", np.sum(nuX))
    if len(nuX.shape) > 1:
        print("reducing nuX to max val")
        nuXo = np.copy(nuX)
        nuX = np.argmax(nuX,axis=-1)
        nuX = np.squeeze(nuX)
        print("new nuX shape, ", nuX.shape)
        maxV = nuXo.shape[-1]
    else:
        nuXo = nuX
    if (saveX):
        X = coords
        if dim == 2:
            print("dim is 2")
            xx = np.zeros((coords.shape[0],3))
            xx[:,0:2] = coords
            xx[:,-1] = z
            X = xx
        np.savez(savename+'_XnuX.npz',X=X,nu_X=nuX)
        vtf.writeVTK(X,[nuX],['geneID'],savename+'_XnuX.vtk',polyData=None)
    
    # random shuffle to do random sample
    if (dim == 2):
        allInfo = np.stack((coords[:,0],coords[:,1],nuX),axis=-1)
        np.random.shuffle(allInfo)
        coords = allInfo[:,0:2]
        nuX = allInfo[:,-1]
    elif (dim == 3):
        allInfo = np.stack((coords[:,0],coords[:,1],coords[:,2],nuX),axis=-1)
        print("all Info shape, ", allInfo.shape)
        np.random.shuffle(allInfo)
        coords = allInfo[:,0:3]
        nuX = allInfo[:,-1]
    
    # divide into cubes of size sig*alpha
    if (dim == 2):
        print("dim is 2")
        coords_labels = np.floor((coords - np.floor(np.min(coords,axis=0)))/(sig*alpha)).astype(int) # minimum number of cubes in x and y 
        totCubes = (np.max(coords_labels[:,0])+1)*(np.max(coords_labels[:,1])+1)
    elif (dim == 3):
        coords_labels = np.floor((coords - np.floor(np.min(coords,axis=0)))/(sig*alpha)).astype(int) # minimum number of cubes in x and y
        print("coords_labels shape, ", coords_labels.shape)
        print("coords_labels max and min, ", np.max(coords_labels,axis=0))
        totCubes = (np.max(coords_labels[:,0])+1)*(np.max(coords_labels[:,1])+1)*(np.max(coords_labels[:,2])+1)
    
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
    uInds,sample,inv,counts = np.unique(coords_labels,return_index=True,return_inverse=True,return_counts=True,axis=0) # returns first occurrence of each cube 
    we = np.sum(nuXo,axis=-1)
    bins = np.arange(np.max(inv) + 2)
    bins = bins - 0.5
    bins = list(bins)
    hist,be = np.histogram(inv,bins=bins,weights=we,density=False)
    print("confirm same shape as cubes")
    print(len(uInds))
    print(hist.shape)
    
    nuZ = nuX[sample] # returns first occurrence of mRNA in cube 
    print("nuZ and mass should be same dim")
    print(nuZ.shape)
    
    Z = coords[np.squeeze(sample),...]
    print("Z shape, ", Z.shape)
    if dim == 2:
        print("dim is 2")
        zz = np.zeros((Z.shape[0],3))
        zz[:,0:2] = Z
        zz[:,-1] = z
        Z = zz
    
    #nu_Z = makeOneHot(nuZ,maxVal=maxV,zeroBased=zeroBased)*coords.shape[0]/nuZ.shape[0] # weigh points with appropriate mass and make large array
    nu_Z = makeOneHot(nuZ,maxVal=maxV,zeroBased=zeroBased)*np.squeeze(hist)[...,None] #*counts[...,None] # weigh particles based on number of MRNA in each cube
    print("sume before :" + str(np.sum(nu_Z)))
    nu_Z = nu_Z*np.sum(nuXo)/np.sum(nu_Z)
    print("sume after :" + str(np.sum(nu_Z)))
    print("nu_Z shape, ", nu_Z.shape)
    maxInd = np.argmax(nu_Z,axis=-1)+1
    vtf.writeVTK(Z,[maxInd,np.sum(nu_Z,axis=-1)],['MAX_VAL_NU','TOTAL_MASS'],savename+'_originalZnu_ZwC' + str(C) + '_sig' + str(sig) + '_discrete.vtk',polyData=None)
    np.savez(savename+'_originalZnu_ZwC' + str(C) + '_sig' + str(sig) + '_discrete.npz',Z=Z, nu_Z=nu_Z)
    if (ztype == 'semi-discrete'):
        nnu_Z,indsToKeep = oneHotMemorySave(nuX,nu_Z,zeroBased=zeroBased)
        nu_Z = nnu_Z + overhead
        nu_Z = reverseOneHot(nu_Z,indsToKeep,maxV)
        # renormalize 
        nu_Z = nu_Z*coords.shape[0]/(np.sum(nu_Z))
        np.savez(savename+'_originalZnu_ZwC' + str(C) + '_sig' + str(sig) + '_semidiscrete_plus' + str(overhead) + '.npz',Z=Z, nu_Z=nu_Z)
        maxInd = np.argmax(nu_Z,axis=-1)+1
        vtf.writeVTK(Z,[maxInd,np.sum(nu_Z,axis=-1)],['MAX_VAL_NU','TOTAL_MASS'],savename+'_originalZnu_ZwC' + str(C) + '_sig' + str(sig) + '_semidiscrete_plus' + str(overhead) + '.vtk',polyData=None)
    elif (ztype == 'uniform'):
        nu_Z[:,:] = 1.0/(nu_Z.shape[-1]) # shape should be number of total values
        nu_Z = nu_Z*np.squeeze(hist)[...,None] #counts[...,None]
        np.savez(savename+'_originalZnu_ZwC' + str(C) + '_sig' + str(sig) + '_uniform.npz',Z=Z, nu_Z=nu_Z)
        maxInd = np.argmax(nu_Z,axis=-1)+1
        vtf.writeVTK(Z,[maxInd,np.sum(nu_Z,axis=-1)],['MAX_VAL_NU','TOTAL_MASS'],savename+'_originalZnu_ZwC' + str(C) + '_sig' + str(sig) + '_uniform.vtk',polyData=None)
    
    print("sum of nu_Z, ", np.sum(nu_Z))
    print("sum of nu_X, ", np.sum(nuXo))
    
    return


def makeSubsampleStratifiedGeneral(Xfile,sig,savename,xtype='discrete',maxV=702,Co=1.2,dim=3,z=None,saveX=True,zeroBased=True,alpha=0.5):
    '''
    Sampling of same types as in makeSubsample, but with weighting geographically in subset
    
    Algorithm:
        - Divide bounding box into squares (cubes) of area / volume sigma^dim
        - Select np.round(C*np.log(counts + 1)) particles from each square (cube) based on counts in the cube 
        
    coordName = name of variable storing coordinates (assume N x 2/3)
    nuName = name of variable storing feature value (assume N x 1)
    
    ztype \in {'discrete', 'semi-discrete', 'uniform'}
    
    '''
    
    ### KATIE NEEDS TO FINISH --> 5/18/23 ###
    C = ((1.0/alpha)**3) * Co
    if (zeroBased is None):
        if (coordName == 'geneInd'):
            zeroBased = True
            print("features are zero-based")
        else:
            zeroBased = False
    info = np.load(Xfile)
    print(info.files)
    coords = info[info.files[0]]
    print("shape of coords")
    if z is None:
        z = np.mean(coords[:,-1])
    if coords.shape[-1] > dim:
        coords = coords[:,0:dim]
    
    print(coords.shape)
    nuX = np.squeeze(info[info.files[1]])
    print("nuX shape, ", nuX.shape)
    print("sum of nuX, ", np.sum(nuX))
    if len(nuX.shape) > 1:
        print("reducing nuX to max val")
        nuXo = np.copy(nuX)
        nuX = np.argmax(nuX,axis=-1)
        nuX = np.squeeze(nuX)
        print("new nuX shape, ", nuX.shape)
        maxV = nuXo.shape[-1]
    else:
        nuXo = nuX
    if (saveX):
        X = coords
        if dim == 2:
            print("dim is 2")
            xx = np.zeros((coords.shape[0],3))
            xx[:,0:2] = coords
            xx[:,-1] = z
            X = xx
        np.savez(savename+'_XnuX.npz',X=X,nu_X=nuX)
        vtf.writeVTK(X,[nuX],['geneID'],savename+'_XnuX.vtk',polyData=None)
    
    # random shuffle to do random sample
    if (dim == 2):
        allInfo = np.stack((coords[:,0],coords[:,1],nuX),axis=-1)
        np.random.shuffle(allInfo)
        coords = allInfo[:,0:2]
        nuX = allInfo[:,-1]
    elif (dim == 3):
        allInfo = np.stack((coords[:,0],coords[:,1],coords[:,2],nuX),axis=-1)
        print("all Info shape, ", allInfo.shape)
        np.random.shuffle(allInfo)
        coords = allInfo[:,0:3]
        nuX = allInfo[:,-1]
    
    # divide into cubes of size sig*alpha
    if (dim == 2):
        print("dim is 2")
        coords_labels = np.floor((coords - np.floor(np.min(coords,axis=0)))/(sig*alpha)).astype(int) # minimum number of cubes in x and y 
        totCubes = (np.max(coords_labels[:,0])+1)*(np.max(coords_labels[:,1])+1)
    elif (dim == 3):
        coords_labels = np.floor((coords - np.floor(np.min(coords,axis=0)))/(sig*alpha)).astype(int) # minimum number of cubes in x and y
        print("coords_labels shape, ", coords_labels.shape)
        print("coords_labels max and min, ", np.max(coords_labels,axis=0))
        totCubes = (np.max(coords_labels[:,0])+1)*(np.max(coords_labels[:,1])+1)*(np.max(coords_labels[:,2])+1)
    
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
    uInds,sample,inv,counts = np.unique(coords_labels,return_index=True,return_inverse=True,return_counts=True,axis=0) # returns first occurrence of each cube 
    we = np.sum(nuXo,axis=-1)
    bins = np.arange(np.max(inv) + 2)
    bins = bins - 0.5
    bins = list(bins)
    hist,be = np.histogram(inv,bins=bins,weights=we,density=False)
    print("confirm same shape as cubes")
    print(len(uInds))
    print(hist.shape)
    
    nuZ = nuX[sample] # returns first occurrence of mRNA in cube 
    print("nuZ and mass should be same dim")
    print(nuZ.shape)
    
    Z = coords[np.squeeze(sample),...]
    print("Z shape, ", Z.shape)
    if dim == 2:
        print("dim is 2")
        zz = np.zeros((Z.shape[0],3))
        zz[:,0:2] = Z
        zz[:,-1] = z
        Z = zz
    
    #nu_Z = makeOneHot(nuZ,maxVal=maxV,zeroBased=zeroBased)*coords.shape[0]/nuZ.shape[0] # weigh points with appropriate mass and make large array
    nu_Z = makeOneHot(nuZ,maxVal=maxV,zeroBased=zeroBased)*np.squeeze(hist)[...,None] #*counts[...,None] # weigh particles based on number of MRNA in each cube
    print("sume before :" + str(np.sum(nu_Z)))
    nu_Z = nu_Z*np.sum(nuXo)/np.sum(nu_Z)
    print("sume after :" + str(np.sum(nu_Z)))
    print("nu_Z shape, ", nu_Z.shape)
    maxInd = np.argmax(nu_Z,axis=-1)+1
    vtf.writeVTK(Z,[maxInd,np.sum(nu_Z,axis=-1)],['MAX_VAL_NU','TOTAL_MASS'],savename+'_originalZnu_ZwC' + str(C) + '_sig' + str(sig) + '_discrete.vtk',polyData=None)
    np.savez(savename+'_originalZnu_ZwC' + str(C) + '_sig' + str(sig) + '_discrete.npz',Z=Z, nu_Z=nu_Z)
    if (ztype == 'semi-discrete'):
        nnu_Z,indsToKeep = oneHotMemorySave(nuX,nu_Z,zeroBased=zeroBased)
        nu_Z = nnu_Z + overhead
        nu_Z = reverseOneHot(nu_Z,indsToKeep,maxV)
        # renormalize 
        nu_Z = nu_Z*coords.shape[0]/(np.sum(nu_Z))
        np.savez(savename+'_originalZnu_ZwC' + str(C) + '_sig' + str(sig) + '_semidiscrete_plus' + str(overhead) + '.npz',Z=Z, nu_Z=nu_Z)
        maxInd = np.argmax(nu_Z,axis=-1)+1
        vtf.writeVTK(Z,[maxInd,np.sum(nu_Z,axis=-1)],['MAX_VAL_NU','TOTAL_MASS'],savename+'_originalZnu_ZwC' + str(C) + '_sig' + str(sig) + '_semidiscrete_plus' + str(overhead) + '.vtk',polyData=None)
    elif (ztype == 'uniform'):
        nu_Z[:,:] = 1.0/(nu_Z.shape[-1]) # shape should be number of total values
        nu_Z = nu_Z*np.squeeze(hist)[...,None] #counts[...,None]
        np.savez(savename+'_originalZnu_ZwC' + str(C) + '_sig' + str(sig) + '_uniform.npz',Z=Z, nu_Z=nu_Z)
        maxInd = np.argmax(nu_Z,axis=-1)+1
        vtf.writeVTK(Z,[maxInd,np.sum(nu_Z,axis=-1)],['MAX_VAL_NU','TOTAL_MASS'],savename+'_originalZnu_ZwC' + str(C) + '_sig' + str(sig) + '_uniform.vtk',polyData=None)
    
    print("sum of nu_Z, ", np.sum(nu_Z))
    print("sum of nu_X, ", np.sum(nuXo))
    
    return

def selectGeneSet(fils,geneInds,geneList,savedir,suff,addone=0):
    '''
    fils = list of files with X, nu_X (zero based) or Z, nu_Z
    geneInds = list of integer indices of columns to select 
    
    Example: selectGeneSet(filsMerfish,[331, 386,  18, 419, 615, 291, 425, 518, 577, 254, 355, 689,  19,
       452, 392, 402, 627, 690, 671, 376],['Dipk1b', 'Grin2a', 'Ank1', 'Kcng1', 'Sorcs3', 'Cnih3', 'Kirrel3',
       'Pcdh8', 'Rph3a', 'Caln1', 'Fndc5', 'Whrn', 'Ankrd6', 'Mdga1',
       'Gucy1a1', 'Hs6st3', 'Stum', 'Wipf3', 'Trp53i11', 'Gfap'],"/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenMerfish/XnuX_Aligned/top20MI/","_lowToHighMI.npz")
       
       selectGeneSet(filsBarSeq,[89,97,91,3,33,53,44,63,105,60,83,103,26,92,84,61,6,65,75,76,64,1,49,47,101,111,56,28],['Rcan2', 'Rora', 'Gpc5', 'Enpp2', 'Tle4', 'Tmsb10', 'Dab1',
       'Ncam2', 'Gnb4', 'Etv1', 'Prkca', 'Hcn1', 'Lpp', 'Lmo4', 'Ncald',
       'Nnat', 'Slc24a3', 'Timp2', 'Pam', 'Slc24a2', 'Alcam', 'Rasgrf2',
       'Camk4', 'Nrsn1', 'Dgkb', 'Slc17a7', 'Gria1', 'Rab3c'],"/cis/home/kstouff4/Documents/MeshRegistration/Particles/BarSeq/top38MI/","_lowToHighMI", addone=1)
    '''
    for f in fils:
        info = np.load(f)
        nu = np.squeeze(info[info.files[1]])
        if len(nu.shape) > 1:
            nuGenes = nu[:,geneInds]
        else:
            nuGenes = np.zeros((nu.shape[0],len(geneInds))).astype('bool')
            for g in range(len(geneInds)):
                nuGenes[:,g] = nu == geneInds[g]
                    
        # eliminate locations with no genes
        keep = np.sum(nuGenes,axis=-1) > 0
        nuGenes = nuGenes[np.squeeze(keep),:]
        X = info[info.files[0]][np.squeeze(keep),:]
        print(X.shape)
        
        np.savez(savedir + f.split('/')[-1].replace('.npz',suff+'.npz'),X=X,nu_X=nuGenes,geneInds=geneInds,geneList=geneList)
        writeParticleVTK(savedir + f.split('/')[-1].replace('.npz',suff+'.npz'),False,geneList)
        
    np.savez(savedir + 'geneList.npz',geneInds=geneInds,geneList=geneList)
    return
        

        


    

