#!/bin/bash

outDir="/cis/home/kstouff4/Documents/MeshRegistration/Particles/SubsampledAllen10umAtlas/"
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
optMethod="Adam"

outPath="/cis/home/kstouff4/Documents/MeshRegistration/Particles/SubsampledAllen10umAtlas/"
pref="_0-50_"
outPath1="${outDir}/Sub$pref"
python3 -c "import estimateSubsampleByLabelScratch as ess; ess.project3D('${outPath1}XnuX.npz',$sig, $nb_iter0, $nb_iter1,'$outPath1',$Nmax,$Npart,maxV=673,optMethod='$optMethod');quit()" >> $outPath1"_outputmem.txt"

pref="_50-100_"
outPath1="${outDir}/Sub$pref"
python3 -c "import estimateSubsampleByLabelScratch as ess; ess.project3D('${outPath1}XnuX.npz',$sig, $nb_iter0, $nb_iter1,'$outPath1',$Nmax,$Npart,maxV=673,optMethod='$optMethod');quit()" >> $outPath1"_outputmem.txt"

pref="_100-150_"
outPath1="${outDir}/Sub$pref"
python3 -c "import estimateSubsampleByLabelScratch as ess; ess.project3D('${outPath1}XnuX.npz',$sig, $nb_iter0, $nb_iter1,'$outPath1',$Nmax,$Npart,maxV=673,optMethod='$optMethod');quit()" >> $outPath1"_outputmem.txt"

pref="_150-200_"
outPath1="${outDir}/Sub$pref"
python3 -c "import estimateSubsampleByLabelScratch as ess; ess.project3D('${outPath1}XnuX.npz',$sig, $nb_iter0, $nb_iter1,'$outPath1',$Nmax,$Npart,maxV=673,optMethod='$optMethod');quit()" >> $outPath1"_outputmem.txt"

pref="_200-250_"
outPath1="${outDir}/Sub$pref"
python3 -c "import estimateSubsampleByLabelScratch as ess; ess.project3D('${outPath1}XnuX.npz',$sig, $nb_iter0, $nb_iter1,'$outPath1',$Nmax,$Npart,maxV=673,optMethod='$optMethod');quit()" >> $outPath1"_outputmem.txt"

pref="_250-300_"
outPath1="${outDir}/Sub$pref"
python3 -c "import estimateSubsampleByLabelScratch as ess; ess.project3D('${outPath1}XnuX.npz',$sig, $nb_iter0, $nb_iter1,'$outPath1',$Nmax,$Npart,maxV=673,optMethod='$optMethod');quit()" >> $outPath1"_outputmem.txt"

pref="_300-350_"
outPath1="${outDir}/Sub$pref"
python3 -c "import estimateSubsampleByLabelScratch as ess; ess.project3D('${outPath1}XnuX.npz',$sig, $nb_iter0, $nb_iter1,'$outPath1',$Nmax,$Npart,maxV=673,optMethod='$optMethod');quit()" >> $outPath1"_outputmem.txt"

pref="_350-400_"
outPath1="${outDir}/Sub$pref"
python3 -c "import estimateSubsampleByLabelScratch as ess; ess.project3D('${outPath1}XnuX.npz',$sig, $nb_iter0, $nb_iter1,'$outPath1',$Nmax,$Npart,maxV=673,optMethod='$optMethod');quit()" >> $outPath1"_outputmem.txt"

pref="_400-450_"
outPath1="${outDir}/Sub$pref"
python3 -c "import estimateSubsampleByLabelScratch as ess; ess.project3D('${outPath1}XnuX.npz',$sig, $nb_iter0, $nb_iter1,'$outPath1',$Nmax,$Npart,maxV=673,optMethod='$optMethod');quit()" >> $outPath1"_outputmem.txt"

pref="_450-500_"
outPath1="${outDir}/Sub$pref"
python3 -c "import estimateSubsampleByLabelScratch as ess; ess.project3D('${outPath1}XnuX.npz',$sig, $nb_iter0, $nb_iter1,'$outPath1',$Nmax,$Npart,maxV=673,optMethod='$optMethod');quit()" >> $outPath1"_outputmem.txt"

pref="_500-550_"
outPath1="${outDir}/Sub$pref"
python3 -c "import estimateSubsampleByLabelScratch as ess; ess.project3D('${outPath1}XnuX.npz',$sig, $nb_iter0, $nb_iter1,'$outPath1',$Nmax,$Npart,maxV=673,optMethod='$optMethod');quit()" >> $outPath1"_outputmem.txt"

pref="_550-600_"
outPath1="${outDir}/Sub$pref"
python3 -c "import estimateSubsampleByLabelScratch as ess; ess.project3D('${outPath1}XnuX.npz',$sig, $nb_iter0, $nb_iter1,'$outPath1',$Nmax,$Npart,maxV=673,optMethod='$optMethod');quit()" >> $outPath1"_outputmem.txt"

pref="_600-650_"
outPath1="${outDir}/Sub$pref"
python3 -c "import estimateSubsampleByLabelScratch as ess; ess.project3D('${outPath1}XnuX.npz',$sig, $nb_iter0, $nb_iter1,'$outPath1',$Nmax,$Npart,maxV=673,optMethod='$optMethod');quit()" >> $outPath1"_outputmem.txt"

pref="_650-700_"
outPath1="${outDir}/Sub$pref"
#python3 -c "import estimateSubsampleByLabelScratch as ess; ess.project3D('${outPath}${pref}XnuX.npz',$sig, $nb_iter0, $nb_iter1,'$outPath1',$Nmax,$Npart,maxV=351);quit()" >> $outPath1"_outputmem.txt"

pref="_700-750_"
outPath1="${outDir}/Sub_$pref"
#python3 -c "import estimateSubsampleByLabelScratch as ess; ess.project3D('${outPath}${pref}XnuX.npz',$sig, $nb_iter0, $nb_iter1,'$outPath1',$Nmax,$Npart,maxV=351);quit()" >> $outPath1"_outputmem.txt"

pref="_1200-1300_"
outPath1="${outDir}/Sub_$pref"
#python3 -c "import estimateSubsampleByLabelScratch as ess; ess.project3D('${outPath}${pref}XnuX.npz',$sig, $nb_iter0, $nb_iter1,'$outPath1',$Nmax,$Npart,maxV=351);quit()" >> $outPath1"_outputmem.txt"

python3 -c "import estimateSubsample as ess; ess.compileZ('$outDir/',pref='optimal'); quit()"

pref="Total_"
outPath1="${outDir}/Sub$pref"
python3 -c "import estimateSubsampleByLabelScratch as ess; ess.project3D('${outPath1}XnuX.npz',$sig, $nb_iter0, $nb_iter1,'$outPath1',$Nmax,$Npart,maxV=673);quit()" >> $outPath1"_outputmem.txt"


outPath="${outDir}Sub_500-600_"
#python3 -c "import estimateSubsampleByLabelScratch as ess; ess.project3D('${outPath}XnuX.npz',$sig, $nb_iter0, $nb_iter1,'$outPath',$Nmax,$Npart);quit()" >> $outPath"_output.txt"

outPath="${outDir}Sub_600-700_"
#python3 -c "import estimateSubsampleByLabelScratch as ess; ess.project3D('${outPath}XnuX.npz',$sig, $nb_iter0, $nb_iter1,'$outPath',$Nmax,$Npart);quit()" >> $outPath"_output.txt"

outPath="${outDir}Sub_700-800_"
#python3 -c "import estimateSubsampleByLabelScratch as ess; ess.project3D('${outPath}XnuX.npz',$sig2, $nb_iter0, $nb_iter1,'$outPath',$Nmax,$Npart);quit()" >> $outPath"_output.txt"

outPath="${outDir}Sub_800-900_"
#python3 -c "import estimateSubsampleByLabelScratch as ess; ess.project3D('${outPath}XnuX.npz',$sig2, $nb_iter0, $nb_iter1,'$outPath',$Nmax,$Npart);quit()" >> $outPath"_output.txt"

outPath="${outDir}Sub_900-1000_"
#python3 -c "import estimateSubsampleByLabelScratch as ess; ess.project3D('${outPath}XnuX.npz',$sig2, $nb_iter0, $nb_iter1,'$outPath',$Nmax,$Npart);quit()" >> $outPath"_output.txt"

outPath="${outDir}Sub_1000-1100_"
#python3 -c "import estimateSubsampleByLabelScratch as ess; ess.project3D('${outPath}XnuX.npz',$sig2, $nb_iter0, $nb_iter1,'$outPath',$Nmax,$Npart);quit()" >> $outPath"_output.txt"

outPath="${outDir}Sub_1100-1200_"
#python3 -c "import estimateSubsampleByLabelScratch as ess; ess.project3D('${outPath}XnuX.npz',$sig2, $nb_iter0, $nb_iter1,'$outPath',$Nmax,$Npart);quit()" >> $outPath"_output.txt"

outPath="${outDir}Sub_1200-1300_"
#python3 -c "import estimateSubsampleByLabelScratch as ess; ess.project3D('${outPath}XnuX.npz',$sig2, $nb_iter0, $nb_iter1,'$outPath',$Nmax,$Npart);quit()" >> $outPath"_output.txt"
