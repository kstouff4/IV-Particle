import numpy as np
import glob

from sys import path as sys_path

sys_path.append('/cis/home/kstouff4/Documents/SurfaceTools/')
import vtkFunctions as vtf



def main():
    '''
    Read in particle approximations and rotate, translate, and scale based on pre-computed manual alignment
    '''
    transLoc = '/cis/home/kstouff4/Documents/SpatialTranscriptomics/Mouse/Mouse1_20220506/zipfiles1/'
    fils = glob.glob('/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenMerfish/ZApprox_sig0.05Uniform/*Znu_ZAll*npz')
    outDir = '/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenMerfish/ZApprox_sig0.05Uniform_Aligned/'
    
    imageNames = ['MASS','MAXVAL_GENE','ENTROPY']
    
    for f in fils:
        numF = f.split('/')[-1].split('_')[1]
        t = glob.glob(transLoc + '*60988' + numF + '*/T_Ri_Rf_um.npz')
        if len(t) < 1:
            print("No transformation file found for " + f)
            continue
        partApprox = np.load(f)
        Z = partApprox['Z']
        trans = np.load(t[0])
        T = trans['T']/1000.0
        Ri = trans['Ri']
        Rf = trans['Rf']
        
        Zn = Z[:,0:2] + T.T
        Zn = Zn@Ri.T
        Zn = Zn@Rf.T
        Znew = Z
        Znew[:,0:2] = Zn
        Znew[:,-1] = Znew[:,-1]/1000.0
        
        fbase = f.split('/')[-1]
        np.savez(outDir + fbase,Z=Znew,nu_Z=partApprox['nu_Z'],h=partApprox['h'])
        mass = np.sum(partApprox['nu_Z'],axis=-1)
        maxval = np.argmax(partApprox['nu_Z'],axis=-1)
        ent = partApprox['h']
        
        vtf.writeVTK(Znew,[mass,maxval,ent],imageNames,outDir+fbase.replace('npz','vtk'),polyData=None)
        

if __name__ == "__main__":
    main()