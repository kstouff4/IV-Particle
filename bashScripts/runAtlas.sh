#!/bin/bash

cd /cis/home/kstouff4/Documents/MeshRegistration/Scripts-KMS/approxCode

nb_iter0=7
nb_iter1=20
Nmax=1500.0
Npart=2000.0
optMethod="LBFGS"
sigma=0.2

atlasImage="/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um/Final/sig0.1__XnuX_669labs.npz"
targetImage="/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um/Final/sig0.1_669_originalZnu_ZwC1.0_sig0.2_uniform.npz"
outPath="/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um/Final/"
maxV=669 

#python3 -c "import subsampleFunctions as sf; sf.makeSubsampleStratified('$atlasImage','X','nu_X',$sigma,'${outPath}sig0.1_669',xtype='discrete',ztype='uniform',overhead=0.1,maxV=$maxV,C=1.0,dim=3,saveX=True,zeroBased=True);quit()"

python3 -c "import approximateFine as ess; ess.project3D('$atlasImage',$sigma, $nb_iter0, $nb_iter1,'${outPath}downFromOld_',$Nmax,$Npart,Zfile='$targetImage',maxV=$maxV,optMethod='$optMethod',C=1.0);quit()" >> "${outPath}downFromOld.txt"

echo $(date) >> "${outPath}downFromOld.txt"

