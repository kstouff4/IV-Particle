import glob 
import numpy as np
from sys import path as sys_path

sys_path.append('/cis/home/kstouff4/Documents/SurfaceTools/')
import vtkFunctions as vtf


def main():
    allenMerfishSlices = '/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenMerfish/ZApprox_sig0.05Uniform_Aligned/'
    barSeqSlices = '/cis/home/kstouff4/Documents/SpatialTranscriptomics/BarSeq/Genes/'
    
    allenGenes = np.load('/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenMerfish/geneList.npz',allow_pickle=True)
    barGenes=np.load('/cis/home/kstouff4/Documents/SpatialTranscriptomics/BarSeq/Genes/geneList.npz',allow_pickle=True)
    aGenes = allenGenes['geneList']
    bGenes = barGenes['genes']
    
    cGenes = []
    indexA = []
    indexB = []
    
    for i in range(len(bGenes)):
        b = bGenes[i]
        if (b in aGenes):
            cGenes.append(b)
            indexB.append(i)
            indexA.append(np.argmax(aGenes == b))
    genes=np.asarray(cGenes)
    indexA = np.asarray(indexA)
    indexB = np.asarray(indexB)
    np.savez('/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenMerfishBarSeq/geneListCommon.npz',genes=genes,indexA=indexA,indexB=indexB)
    '''
    aSlices = glob.glob(allenMerfishSlices + '*.npz')
    totZ = []
    totnuZ = []
    totPart = 0
    for a in aSlices:
        info = np.load(a) # Z, nu_Z, h
        totZ.append(info['Z'])
        totnuZ.append(info['nu_Z'][:,indexA])
        totPart += len(info['Z'])
    
    geneNum = totnuZ[0].shape[-1]
    totnuZa = np.zeros((totPart,geneNum))
    totZa = np.zeros((totPart,totZ[0].shape[-1]))
    ct = 0
    for i in range(len(totnuZ)):
        ct2 = ct + len(totZ[i])
        totZa[ct:ct2,:] = totZ[i]
        totnuZa[ct:ct2,:] = totnuZ[i]
        ct = ct2
    np.savez('/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenMerfishBarSeq/allenCommonGenes50um.npz',Z=totZa,nu_Z=totnuZa)
    cGenes.append('maxVal')
    geneVals = []
    for j in range(len(cGenes)-1):
        geneVals.append(totnuZa[:,j])
    geneVals.append(np.argmax(totnuZa,axis=-1))
    cGenes.append('totalMass')
    geneVals.append(np.sum(totnuZa,axis=-1))
    vtf.writeVTK(totZa,geneVals,cGenes,'/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenMerfishBarSeq/allenCommonGenes50um.vtk',polyData=None)
    '''
    bSlices = glob.glob(barSeqSlices + 'slice*.npz')
    totZ = []
    totnuZ = []
    totPart = 0
    for b in bSlices:
        info = np.load(b) # X, nu_X, nu_Xc
        nuXs = info['nu_X']-1 # 1 based 
        nuXoh = np.zeros((nuXs.shape[0],len(bGenes)))
        nuXoh[np.arange(nuXoh.shape[0]),np.squeeze(nuXs)] = 1
        nuXs = nuXoh[:,indexB]
        indsNZ = np.sum(nuXs,axis=-1) > 0
        totZ.append(info['X'][indsNZ,...])
        totnuZ.append(nuXs[indsNZ,...])
        totPart += np.sum(indsNZ)
        
    geneNum = totnuZ[0].shape[-1]
    totnuZb = np.zeros((totPart,geneNum))
    totZb = np.zeros((totPart,totZ[0].shape[-1]))
    ct = 0
    for i in range(len(totnuZ)):
        ct2 = ct + len(totZ[i])
        totZb[ct:ct2,:] = totZ[i]
        totnuZb[ct:ct2,:] = totnuZ[i]
        ct = ct2
    np.savez('/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenMerfishBarSeq/barSeqCommonGenesorig.npz',Z=totZb,nu_Z=totnuZb)
    geneVals = []
    for j in range(len(cGenes)):
        geneVals.append(totnuZb[:,j])
    geneVals.append(np.argmax(totnuZb,axis=-1))
    geneVals.append(np.sum(totnuZb,axis=-1))
    cGenes.append('maxVal')
    cGenes.append('totalMass')
    vtf.writeVTK(totZb,geneVals,cGenes,'/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenMerfishBarSeq/barSeqCommonGenesorig.vtk',polyData=None)
    
    return

if __name__ == "__main__":
    main()