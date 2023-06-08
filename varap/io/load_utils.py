'''
Class of generic functions for altering format of input data.
I/O: npz files
'''

# Author: Kaitlin Stouffer (kstouff4@jhmi.edu)

def centerAndScale(filename,keepZ=False,s=0.001):
    '''
    Center data around center of Mass and then Scale coordinates
    '''
    data = np.load(filename,allow_pickle=True)
    X = data[data.files[0]]
    c = np.mean(X,axis=0)
    X = X - c
    if keepZ:
        z = c[-1]
        X[:,-1] = z
    X = X*s
    di = dict(data)
    di[data.files[0]] = X
    newName = filename.replace('.npz','_centeredAndScaled' + str(s) + '.npz')
    np.savez(newName,**di)
    return newName