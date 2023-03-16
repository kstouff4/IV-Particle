#!/bin/bash

# Call with desired folder and string to search (e.g. ./getDensity.sh /cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenMerfish/ZApprox_sig0.05Uniform/ optimalZnu_ZAll Z nu_Z 0.05)

cd /cis/home/kstouff4/Documents/MeshRegistration/ParticleLDDMMQP/sandbox/

inpDir=$1
suff=$2
x=$3
nux=$4
sig=$5


fils=$(ls $inpDir | grep $suff | grep npz)
for f in ${fils[*]}; do
    python3 -c "import analyzeOutput as ao; import numpy as np; x = np.load('$f'); X=x['$x']; nuX = x['nux']; ao.getLocalDensity(X,nuX,$sig,('$f').replace('.npz','_density.vtk')); quit()"
done
