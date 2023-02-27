#!/bin/bash

cd /cis/home/kstouff4/Documents/MeshRegistration/Scripts-KMS/approxCode

nb_iter0=3
nb_iter1=7
Nmax=1500.0
Npart=2000.0
optMethod="LBFGS"
sigma=0.025
marg=2.0

outPath="/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenMerfish"
outPathX=$outPath'/XSplits/'
outPathZ=$outPath"/ZApprox-XComb_sig${sigma}_Feb20/"
maxV=702
outDir=$outPath"/ZApprox-XComb_sig${sigma}_Feb20_Stitched/"
mkdir $outDir

pref='_201_'
python3 -c "import smootheBoundaries as sb; sb.stitchAllQuadrants('$outPathX','$outPathZ','$pref',$sigma,$marg,$outDir,$nb_iter0,$nb_iter1,$Nmax,$Npart,$maxV);quit()" >> $outDir$pref"Stitched.txt"


