import numpy as np

def writeVTK(YXZ,features,featureNames,savename,polyData=None,fields=None,fieldNames=None):
    '''
    Write YXZ coordinates (assume numpts x 3 as X,Y,Z in vtk file)
    polyData should be in format 3 vertex, vertex, vertex (0 based numbering)
    '''
    f_out_data = []

    # Version 3.0 header
    f_out_data.append("# vtk DataFile Version 3.0\n")
    f_out_data.append("Surface Data\n")
    f_out_data.append("ASCII\n")
    f_out_data.append("DATASET POLYDATA\n")
    f_out_data.append("\n")

    num_pts = YXZ.shape[0]
    f_out_data.append("POINTS %d float\n" % num_pts)
    for pt in range(num_pts):
        f_out_data.append("%f %f %f\n" % (YXZ[pt,1], YXZ[pt,0], YXZ[pt,2])) # x,y,z
    
    if (polyData is not None):
        r = polyData.shape[0]
        c = polyData.shape[1]
        if (c > 3):
            f_out_data.append("POLYGONS %d %d\n" % (r,c*r))
        else:
            f_out_data.append("LINES %d %d\n" % (r,c*r))
        for i in range(r):
            if (c == 4):
                f_out_data.append("%d %d %d %d\n" % (polyData[i,0], polyData[i,1], polyData[i,2], polyData[i,3]))
            elif (c == 3):
                f_out_data.append("%d %d %d\n" % (polyData[i,0], polyData[i,1], polyData[i,2]))
    f_out_data.append("POINT_DATA %d\n" % num_pts)
    fInd = 0;
    for f in featureNames:
        f_out_data.append("SCALARS " + f + " float 1\n")
        f_out_data.append("LOOKUP_TABLE default\n")
        fCurr = features[fInd]
        for pt in range(num_pts):
            f_out_data.append("%.9f\n" % fCurr[pt])
        fInd = fInd + 1
    if fields is not None:
        for f in range(len(fields)):
            f_out_data.append("VECTORS " + fieldNames[f] + " float\n")
            fieldCurr = fields[f]
            #for pt in range(num_pts):
                

    # Write output data array to file
    with open(savename, "w") as f_out:
        f_out.writelines(f_out_data)
    return

def writeParticleVTK(npzfile):
    x = np.load(npzfile)
    X = x[x.files[0]]
    nuX = x[x.files[1]]
    
    imageNames = ['Weight','Maximum_Feature_Dimension']
    imageVals = [np.sum(nuX,axis=-1),np.argmax(nuX,axis=-1)]
    zetaX = nuX/np.sum(nuX,axis=-1)[...,None]
    for f in range(nuX.shape[-1]):
        imageNames.append('Feature_' + str(f) + '_Probabilities')
        imageVals.append(zetaX[:,f])
    writeVTK(X,imageVals,imageNames,npzfile.replace('.npz','.vtk'))
    return