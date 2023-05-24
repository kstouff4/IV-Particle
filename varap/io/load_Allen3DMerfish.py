import glob
import numpy as np

class Allen3DMerfishLoader:
    
    def __init__(self,rootDir,res,numF=None,deltaF=False):
        '''
        rootDir = directory with XnuX.npz files for each slice
        res = [x,y,z] resolution; x = 0 if finest resolution is unknown
        numF = number of feature dimensions
        deltaF = true if features are encoded with index of feature dimension rather than size numF vector
        '''
        self.filenames = glob.glob(rootDir + '*X.npz')
        self.res = res # x,y,z resolution as list
        if numF is not None:
            self.numFeatures = numF
        else:
            self.numFeatures = None
        
        self.deltaF = deltaF
        self.sizes = None
    
    def getSizes(self):
        '''
        Returns maximum size (particles) of slices and number of features
        
        Sets number of Features and sizes of slices 
        '''
        
        sizes = []
        uniqueF = []
        
        totS = 0
        
        for f in self.filenames:
            n = np.load(f)
            sizes.append(n[n.files[0]].shape[0])
            
            # assume discrete if only one value stored 
            if len(n[n.files[1]].shape) < 2 or n[n.files[1]].shape[1] == 1 or deltaF:
                uniqueF.append(np.unique(n[n.files[1]]))
            
            else:
                if self.numFeatures is None:
                    self.numFeatures = n[n.files[1]].shape[1]
                else:
                    print("Expected Features is ", self.numFeatures)
                    print("Features Read in Dataset is ", n[n.files[1]].shape[1])
        if self.numFeatures is None:
            self.numFeatures = len(np.unique(np.asarray(uniqueF)))
        
        self.sizes = sizes
        return max(self.sizes), self.numFeatures
                
                
    def getSlice(self,index):
        '''
        returns the data corresponding to a single slice only 
        '''
        
        info = np.load(self.filenames[index])
        coordinates = info[info.files[0]]
        features = info[info.files[1]]
        return coordinates, features
    
    def subSample(self,outpath,resolution,uniform=True):
        '''
        subsample each of datasets per file in filenames and write in outpath 
        subsample will be done with given resolution
        
        two choices of sampling: random sampling for initialization or stratified sampling with uniform distribution over features
        '''
        
        return
        

if __name__ == '__main__':
    a = Allen3DMerfishLoader('/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenMerfish/XnuX_Aligned/',[0,0,0.100])
    print("filenames are: ", len(a.filenames))
    particles,features = a.getSizes()
    print(a.sizes)
    print(a.numFeatures)
    
    