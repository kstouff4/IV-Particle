#!/bin/bash

cd /cis/home/kstouff4/Documents/MeshRegistration/Scripts-KMS/approxCode

outPathX="/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenMerfish/XnuX/_201_XnuX._"
sigma=25.0 # micron units 
outPathZ="/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenMerfish/ZApprox-XComb_sig${sigma}/"
mkdir $outPathZ
outPathZ="${outPathZ}_201_"

# Optimization Parameters 
nb_iter0=4
nb_iter1=15
Nmax=5000.0
Npart=1000.0
optMethod="LBFGS"
maxV=702
C=1.0

inds=(0)
sigma=0.025 # assume rescaled 
for i in ${inds[*]}; do
    #echo $(date) >> "${outPathZ}Discrete_C1.0.txt"
    #python3 -c "import estimateSubsampleByLabelScratchTestExperiments as ess; ess.project3D('$outPathX${i}.npz',$sigma, $nb_iter0, $nb_iter1,'${outPathZ}Discrete_',$Nmax,$Npart,Zfile='${outPathZ}originalZnu_ZwC1.0_sig25.0_discrete.npz',maxV=$maxV,optMethod='$optMethod',C=$C);quit()" >> "${outPathZ}Discrete_C1.0.txt"
    #echo $(date) >> "${outPathZ}Discrete_C1.0.txt"
    
    echo $(date) >> "${outPathZ}Semidiscrete_C1.0_${i}.txt"
    python3 -c "import estimateSubsampleByLabelScratchTestExperiments as ess; ess.project3D('$outPathX${i}.npz',$sigma, $nb_iter0, $nb_iter1,'${outPathZ}Semidiscrete_${i}_',$Nmax,$Npart,Zfile='${outPathZ}originalZnu_ZwC1.0_sig25.0_semidiscrete_plus0.05._${i}.npz',maxV=$maxV,optMethod='$optMethod',C=$C);quit()" >> "${outPathZ}Semidiscrete_C1.0_${i}.txt"
    echo $(date) >> "${outPathZ}Semidiscrete_C1.0_${i}.txt"
done

