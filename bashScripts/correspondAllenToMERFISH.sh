#!/bin/bash
cd /cis/home/kstouff4/Documents/MeshRegistration/Scripts-KMS/approxCode

inpathX='/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenMerfish/XSplits/'
outpath='/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenMerfish/Stitched_sig0.025/'
inpathZ='/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenMerfish/ZApprox-XComb_sig0.025/'
inpathX=$outpath
inpathZ=$outpath

mkdir $outpath

maxVal=702
sigma=0.025
margin=2.0
nb_iter0=3
nb_iter1=7
NmaxI=5000.0
NpartI=1000.0
allenNPZ='/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um/Experiments/Stitched/Sub_13slabs_0-1_optimalZnu_ZAllwC1.2_sig0.05_Nmax5000.0_Npart1000.0.npz'

pref="Sub__202"
#python3 -c "import smootheBoundaries as sb; sb.stitchAllQuadrants('$inpathX','$inpathZ','$pref',$sigma,$margin,'$outpath',$nb_iter0,$nb_iter1,$NmaxI,$NpartI,$maxVal); quit()" >> $outpath${pref}_log.txt

x1="${inpathX}Sub__202_01_XnuX.npz"
x2="${inpathX}Sub__202_23_XnuX.npz"
z1="${inpathZ}Sub__202_01_optimalZnu_ZAllwC1.2_sig0.025_Nmax5000.0_Npart1000.0.npz"
z2="${inpathZ}Sub__202_23_optimalZnu_ZAllwC1.2_sig0.025_Nmax5000.0_Npart1000.0.npz"

python3 -c "import smootheBoundaries as sb; sb.stitchQuadrants('$x1','$x2','$z1','$z2','$outpath',$sigma,$nb_iter0, $nb_iter1, margin=2.0,Nmax=5000.0,Npart=1000.0,maxV=702,optMethod='LBFGS'); quit()" >> $outpath${pref}_log.txt

sAllen=-2.763
eAllen=-2.737
python3 -c "import analyzeDistributions as ad; ad.selectPlanes('$allenNPZ',thick=$sigma,ax=2,s=$sAllen,e=$eAllen);quit()"