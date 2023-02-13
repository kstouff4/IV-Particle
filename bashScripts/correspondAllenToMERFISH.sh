#!/bin/bash
cd /cis/home/kstouff4/Documents/MeshRegistration/Scripts-KMS/approxCode

inpathX='/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenMerfish/XSplits/'
outpath='/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenMerfish/Stitched_sig0.025/'
inpathZ='/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenMerfish/ZApprox-XComb_sig0.025/'

mkdir $outpath

maxVal=702
sigma=0.025
margin=2.0
nb_iter0=3
nb_iter1=7
NmaxI=5000.0
NpartI=1000.0

pref="_202"
python3 -c "import smootheBoundaries as sb; sb.stitchAllQuadrants('$inpathX','$inpathZ','$pref',$sigma,$margin,'$outpath',$nb_iter0,$nb_iter1,$NmaxI,$NpartI,$maxVal); quit()" >> $outpath${pref}_log.txt

sAllen=-2.763
eAllen=-2.737
python3 -c "import analyzeDistributions as ad; ad.selectPlanes('$allenNPZ',thick=$sigma,ax=2,s=$sAllen,e=$eAllen);quit()"