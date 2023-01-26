#!/bin/bash
cd /cis/home/kstouff4/Documents/MeshRegistration/Scripts-KMS/approxCode

Xfile="/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um/XSplits/Sub_0200-0300_XnuX._0.npz"
sigma=0.1
savename="/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um/SamplingExperiments/Sub_0200-0300_0"
delta=0.1

python3 -c "import subsampleFunctions as sf; sf.makeSubsample('$Xfile',$sigma,'$savename',xtype='discrete',ztype='semi-discrete',overhead=$delta,maxV=673,C=0.6); quit()"

Xfile="/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um/XSplits/Sub_0200-0300_XnuX._1.npz"
sigma=0.1
savename="/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um/SamplingExperiments/Sub_0200-0300_1"

python3 -c "import subsampleFunctions as sf; sf.makeSubsample('$Xfile',$sigma,'$savename',xtype='discrete',ztype='semi-discrete',overhead=$delta,maxV=673,C=0.6); quit()"

Xfile="/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um/XSplits/Sub_0200-0300_XnuX._2.npz"
sigma=0.1
savename="/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um/SamplingExperiments/Sub_0200-0300_2"

python3 -c "import subsampleFunctions as sf; sf.makeSubsample('$Xfile',$sigma,'$savename',xtype='discrete',ztype='semi-discrete',overhead=$delta,maxV=673,C=0.6); quit()"

Xfile="/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um/XSplits/Sub_0200-0300_XnuX._3.npz"
sigma=0.1
savename="/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um/SamplingExperiments/Sub_0200-0300_3"

python3 -c "import subsampleFunctions as sf; sf.makeSubsample('$Xfile',$sigma,'$savename',xtype='discrete',ztype='semi-discrete',overhead=$delta,maxV=673,C=0.6); quit()"