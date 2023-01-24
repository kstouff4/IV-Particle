#!/bin/bash

outDir="/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um/"
mkdir $outDir
mkdir $outDir"Experiments/"

export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.75'

sig=0.05 # 50 microns
sig2=0.1 # 100 microns
sig3=0.5 # 500 microns (0.5 mm) 
sig4=0.01
nb_iter0=3
nb_iter1=10
Nmax=5000.0
Npart=1000.0
optMethod="LBFGS"

outPath="/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um/"
pref="_600-700_"
outPath1="${outDir}/Sub$pref"
outoutPath="${outPath}/Experiments/Sub$pref"
echo $(date) >> $outoutPath"_outputmem._0_${Nmax}_$Npart.txt"
#python3 -c "import estimateSubsampleByLabelScratchTestExperiments as ess; ess.project3D('${outPath1}XnuX._0.npz',$sig, $nb_iter0, $nb_iter1,'${outoutPath}_0_',$Nmax,$Npart,maxV=673,optMethod='$optMethod');quit()" >> $outoutPath"_outputmem._0_${Nmax}_$Npart.txt"
echo $(date) >> $outoutPath"_outputmem._0_${Nmax}_$Npart.txt"

echo $(date) >> $outoutPath"_outputmem._1_${Nmax}_$Npart.txt"
#python3 -c "import estimateSubsampleByLabelScratchTestExperiments as ess; ess.project3D('${outPath1}XnuX._1.npz',$sig, $nb_iter0, $nb_iter1,'${outoutPath}_1_',$Nmax,$Npart,maxV=673,optMethod='$optMethod');quit()" >> $outoutPath"_outputmem._1_${Nmax}_$Npart.txt"
echo $(date) >> $outPath1"_outputmem._1_${Nmax}_$Npart.txt"

echo $(date) >> $outoutPath"_outputmem._2_${Nmax}_$Npart.txt"
#python3 -c "import estimateSubsampleByLabelScratchTestExperiments as ess; ess.project3D('${outPath1}XnuX._2.npz',$sig, $nb_iter0, $nb_iter1,'${outoutPath}_2_',$Nmax,$Npart,maxV=673,optMethod='$optMethod');quit()" >> $outoutPath"_outputmem._2_${Nmax}_$Npart.txt"
echo $(date) >> $outoutPath"_outputmem._2_${Nmax}_$Npart.txt"

echo $(date) >> $outoutPath"_outputmem._3_${Nmax}_$Npart.txt"
python3 -c "import estimateSubsampleByLabelScratchTestExperiments as ess; ess.project3D('${outPath1}XnuX._3.npz',$sig, $nb_iter0, $nb_iter1,'${outoutPath}_3_',$Nmax,$Npart,maxV=673,optMethod='$optMethod');quit()" >> $outoutPath"_outputmem._3_${Nmax}_$Npart.txt"
echo $(date) >> $outoutPath"_outputmem._3_${Nmax}_$Npart.txt"
