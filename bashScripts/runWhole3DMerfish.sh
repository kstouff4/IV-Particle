#!/bin/bash

cd /cis/home/kstouff4/Documents/MeshRegistration/Scripts-KMS/approxCode

nb_iter0=0
nb_iter1=2
Nmax=1500.0
Npart=2000.0
optMethod="LBFGS"
s0=0.1
s1=0.5
s2=1.0

atlasImage="/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenMerfish/ZnuZ_Aligned/top20MI/sig0.1/All_ZnuZ_sig0.05.npz"
targetImage="/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenMerfish/ZnuZ_Aligned/top20MI/sig0.1/All_ZnuZ_sig0.05_originalZnu_ZwC8.0_sig0.1_uniform.npz"
outPath="/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenMerfish/ZnuZ_Aligned/top20MI/sig0.1/"
maxV=20
mkdir $outPath

#python3 -c "import subsampleFunctions as sf; sf.makeSubsampleStratified('$atlasImage','Z','nu_Z',$s0,'${outPath}All_ZnuZ_sig0.05',xtype='semi-discrete',ztype='uniform',overhead=0.1,maxV=$maxV,Co=1.0,dim=3,saveX=False,zeroBased=True,alpha=0.5);quit()"

python3 -c "import approximateIntermediateMultiScaleNewLoss as ess; ess.project3D('$atlasImage',[$s0], $nb_iter0, $nb_iter1,'${outPath}All_ZnuZ_',$Nmax,$Npart,Zfile='$targetImage',maxV=$maxV,optMethod='$optMethod',C=8.0);quit()" > "${outPath}All_ZnuZ_alpha.txt"

echo $(date) >> "${outPath}All_ZnuZ_alpha.txt"

