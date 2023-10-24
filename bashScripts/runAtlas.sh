#!/bin/bash

cd /cis/home/kstouff4/Documents/MeshRegistration/Scripts-KMS/approxCode

nb_iter0=10
nb_iter1=30
Nmax=1500.0
Npart=2000.0
optMethod="LBFGS"
sigma=0.4
bw=75 # Atlas = 75

atlasImage="/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um/Final/Sub_16slabs_0-1_optimalZnu_ZAllwC1.2_sig0.1_Nmax5000.0_Npart1000.0.npz"
targetImage="/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um/Final/sig0.1_669_originalZnu_ZwC1.0_sig0.2_uniform.npz"
targetImage="/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um/Final/sig0.4__originalZnu_ZwC1.2_sig0.4_semidiscrete.npz"
outPath="/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um/Final/"
maxV=673 

python3 -c "import subsampleFunctions as sf; sf.makeSubsample('$atlasImage',$sigma,'${outPath}sig0.4_',xtype='semi-discrete',ztype='semi-discrete',overhead=0.05,maxV=$maxV,C=1.2);quit()"

echo $(date) > "${outPath}sig0.4_its10-30.txt"

python3 -c "import byBandApproximateIntermediateMultiScaleNewLoss as ess; ess.project3D('$atlasImage',[$sigma], $nb_iter0, $nb_iter1,'${outPath}sig0.4_its10-30_',$Nmax,$Npart,Zfile='$targetImage',maxV=$maxV,optMethod='$optMethod',C=1.2,bw=$bw);quit()" >> "${outPath}sig0.4_its10-30.txt"

echo $(date) >> "${outPath}sig0.4_its10-30.txt"

