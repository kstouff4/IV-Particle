#!/bin/bash

outDir="/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um/"
mkdir $outDir

export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.75'

sig=0.05 # 50 microns
sig2=0.1 # 100 microns
sig3=0.5 # 500 microns (0.5 mm) 
sig4=0.01
nb_iter0=3
nb_iter1=15
Nmax=5000.0
Npart=500.0
optMethod="LBFGS"

outPath="/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um/"
pref="_0-100_"
outPath1="${outDir}/Sub$pref"
echo $(date) >> $outPath1"_outputmem._0.txt"
python3 -c "import estimateSubsampleByLabelScratch as ess; ess.project3D('${outPath1}XnuX._0.npz',$sig, $nb_iter0, $nb_iter1,'${outPath1}_0_',$Nmax,$Npart,maxV=673,optMethod='$optMethod');quit()" >> $outPath1"_outputmem._0.txt"
echo $(date) >> $outPath1"_outputmem._0.txt"

echo $(date) >> $outPath1"_outputmem._1.txt"
python3 -c "import estimateSubsampleByLabelScratch as ess; ess.project3D('${outPath1}XnuX._1.npz',$sig, $nb_iter0, $nb_iter1,'${outPath1}_1',$Nmax,$Npart,maxV=673,optMethod='$optMethod');quit()" >> $outPath1"_outputmem._1.txt"
echo $(date) >> $outPath1"_outputmem._1.txt"

echo $(date) >> $outPath1"_outputmem._2.txt"
python3 -c "import estimateSubsampleByLabelScratch as ess; ess.project3D('${outPath1}XnuX._2.npz',$sig, $nb_iter0, $nb_iter1,'${outPath1}_2',$Nmax,$Npart,maxV=673,optMethod='$optMethod');quit()" >> $outPath1"_outputmem._2.txt"
echo $(date) >> $outPath1"_outputmem._2.txt"

echo $(date) >> $outPath1"_outputmem._3.txt"
python3 -c "import estimateSubsampleByLabelScratch as ess; ess.project3D('${outPath1}XnuX._3.npz',$sig, $nb_iter0, $nb_iter1,'${outPath1}_3',$Nmax,$Npart,maxV=673,optMethod='$optMethod');quit()" >> $outPath1"_outputmem._3.txt"
echo $(date) >> $outPath1"_outputmem._3.txt"

pref="_100-200_"
outPath1="${outDir}/Sub$pref"
echo $(date) >> $outPath1"_outputmem._0.txt"
python3 -c "import estimateSubsampleByLabelScratch as ess; ess.project3D('${outPath1}XnuX._0.npz',$sig, $nb_iter0, $nb_iter1,'${outPath1}_0_',$Nmax,$Npart,maxV=673,optMethod='$optMethod');quit()" >> $outPath1"_outputmem._0.txt"
echo $(date) >> $outPath1"_outputmem._0.txt"

echo $(date) >> $outPath1"_outputmem._1.txt"
python3 -c "import estimateSubsampleByLabelScratch as ess; ess.project3D('${outPath1}XnuX._1.npz',$sig, $nb_iter0, $nb_iter1,'${outPath1}_1',$Nmax,$Npart,maxV=673,optMethod='$optMethod');quit()" >> $outPath1"_outputmem._1.txt"
echo $(date) >> $outPath1"_outputmem._1.txt"

echo $(date) >> $outPath1"_outputmem._2.txt"
python3 -c "import estimateSubsampleByLabelScratch as ess; ess.project3D('${outPath1}XnuX._2.npz',$sig, $nb_iter0, $nb_iter1,'${outPath1}_2',$Nmax,$Npart,maxV=673,optMethod='$optMethod');quit()" >> $outPath1"_outputmem._2.txt"
echo $(date) >> $outPath1"_outputmem._2.txt"

echo $(date) >> $outPath1"_outputmem._3.txt"
python3 -c "import estimateSubsampleByLabelScratch as ess; ess.project3D('${outPath1}XnuX._3.npz',$sig, $nb_iter0, $nb_iter1,'${outPath1}_3',$Nmax,$Npart,maxV=673,optMethod='$optMethod');quit()" >> $outPath1"_outputmem._3.txt"
echo $(date) >> $outPath1"_outputmem._3.txt"

pref="_200-300_"
outPath1="${outDir}/Sub$pref"
echo $(date) >> $outPath1"_outputmem._0.txt"
python3 -c "import estimateSubsampleByLabelScratch as ess; ess.project3D('${outPath1}XnuX._0.npz',$sig, $nb_iter0, $nb_iter1,'${outPath1}_0_',$Nmax,$Npart,maxV=673,optMethod='$optMethod');quit()" >> $outPath1"_outputmem._0.txt"
echo $(date) >> $outPath1"_outputmem._0.txt"

echo $(date) >> $outPath1"_outputmem._1.txt"
python3 -c "import estimateSubsampleByLabelScratch as ess; ess.project3D('${outPath1}XnuX._1.npz',$sig, $nb_iter0, $nb_iter1,'${outPath1}_1',$Nmax,$Npart,maxV=673,optMethod='$optMethod');quit()" >> $outPath1"_outputmem._1.txt"
echo $(date) >> $outPath1"_outputmem._1.txt"

echo $(date) >> $outPath1"_outputmem._2.txt"
python3 -c "import estimateSubsampleByLabelScratch as ess; ess.project3D('${outPath1}XnuX._2.npz',$sig, $nb_iter0, $nb_iter1,'${outPath1}_2',$Nmax,$Npart,maxV=673,optMethod='$optMethod');quit()" >> $outPath1"_outputmem._2.txt"
echo $(date) >> $outPath1"_outputmem._2.txt"

echo $(date) >> $outPath1"_outputmem._3.txt"
python3 -c "import estimateSubsampleByLabelScratch as ess; ess.project3D('${outPath1}XnuX._3.npz',$sig, $nb_iter0, $nb_iter1,'${outPath1}_3',$Nmax,$Npart,maxV=673,optMethod='$optMethod');quit()" >> $outPath1"_outputmem._3.txt"
echo $(date) >> $outPath1"_outputmem._3.txt"





