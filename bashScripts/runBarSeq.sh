#!/bin/bash

cd /cis/home/kstouff4/Documents/MeshRegistration/Scripts-KMS/approxCode

nb_iter0=0 #7
nb_iter1=150 #20
Nmax=1500.0
Npart=2000.0
optMethod="LBFGS"
s0=0.2
s1=0.5
s2=1.0

atlasImage0="/cis/home/kstouff4/Documents/MeshRegistration/Particles/BarSeq/slicesAll_[28-56-111-101-47]_mmRedone_XoneHot.npz"
targetImage="/cis/home/kstouff4/Documents/MeshRegistration/Particles/BarSeq/slicesAll_alpha0.5__originalZnu_ZwC8.0_sig0.2_uniform.npz"
outPath="/cis/home/kstouff4/Documents/MeshRegistration/Particles/BarSeq/"
maxV=5

s0=0.4
atlasImage="/cis/home/kstouff4/Documents/MeshRegistration/Particles/BarSeq/Redo__optimalZnu_ZAllwC8.0_sig[0.2]_Nmax1500.0_Npart2000.0.npz"
targetImage="/cis/home/kstouff4/Documents/MeshRegistration/Particles/BarSeq/slicesAll_alpha0.5__originalZnu_ZwC8.0_sig0.4_uniform.npz"


#python3 -c "import subsampleFunctions as sf; sf.makeSubsampleStratified('$atlasImage0','X','nu_X',$s0,'${outPath}slicesAll_alpha0.5_',xtype='discrete',ztype='uniform',overhead=0.1,maxV=$maxV,Co=1.0,dim=3,saveX=False,zeroBased=True,alpha=0.5);quit()"

python3 -c "import approximateIntermediateMultiScaleNewLoss as ess; ess.project3D('$atlasImage',[$s0], $nb_iter0, $nb_iter1,'${outPath}Redo2_',$Nmax,$Npart,Zfile='$targetImage',maxV=$maxV,optMethod='$optMethod',C=8.0);quit()" > "${outPath}Redo2_genes_sig$s0.txt"

echo $(date) >> "${outPath}Redo2_genes_sig$s0.txt"

