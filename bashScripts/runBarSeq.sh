#!/bin/bash

cd /cis/home/kstouff4/Documents/MeshRegistration/Scripts-KMS/approxCode

nb_iter0=7
nb_iter1=20
Nmax=1500.0
Npart=2000.0
optMethod="LBFGS"
sigma=0.2

atlasImage="/cis/home/kstouff4/Documents/MeshRegistration/Particles/BarSeq/slicesAll_[28-56-111-101-47]_mmRedone_new.npz"
targetImage="/cis/home/kstouff4/Documents/MeshRegistration/Particles/BarSeq/slicesAll_[28-56-111-101-47]_mmRedone_originalZnu_ZwC1.0_sig0.2_uniform_new.npz"
outPath="/cis/home/kstouff4/Documents/MeshRegistration/Particles/BarSeq/"
maxV=5

#python3 -c "import subsampleFunctions as sf; sf.makeSubsampleStratified('$atlasImage','X','nu_X',$sigma,'${outPath}slicesAll_',xtype='discrete',ztype='uniform',overhead=0.1,maxV=$maxV,C=1.0,dim=3,saveX=False,zeroBased=True);quit()"

python3 -c "import approximateFine as ess; ess.project3D('$atlasImage',$sigma, $nb_iter0, $nb_iter1,'${outPath}5mi_',$Nmax,$Npart,Zfile='$targetImage',maxV=$maxV,optMethod='$optMethod',C=1.0);quit()" >> "${outPath}5mi_genes_new.txt"

echo $(date) >> "${outPath}5mi_genes_new.txt"

