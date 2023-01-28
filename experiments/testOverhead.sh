#!/bin/bash

cd /cis/home/kstouff4/Documents/MeshRegistration/Scripts-KMS/approxCode

outPathX="/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenMerfish/_201_XnuX.npz"
sigma=25.0 # micron units 
outPathZ="/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenMerfish/ZApprox-XComb_sig${sigma}/_201_"

# Optimization Parameters 
nb_iter0=5
nb_iter1=20
Nmax=5000.0
Npart=1000.0
optMethod="LBFGS"
maxV=702

inds=(0)

for i in ${inds[*]}; do
    echo $(date) >> "${outPathZ}Discrete_C1.0.txt"
    python3 -c "import estimateSubsampleByLabelScratchTestExperiments as ess; ess.project3D('$outPathX',$sigma, $nb_iter0, $nb_iter1,'${outPathZ}Discrete_',$Nmax,$Npart,Zfile='${outPathZ}originalZnu_ZwC1.0_sig25.0_discrete.npz',maxV=$maxV,optMethod='$optMethod');quit()" >> "${outPathZ}Discrete_C1.0.txt"
    echo $(date) >> "${outPathZ}Discrete_C1.0.txt"
    
    echo $(date) >> "${outPathZ}Semidiscrete_C1.0.txt"
    python3 -c "import estimateSubsampleByLabelScratchTestExperiments as ess; ess.project3D('$outPathX',$sigma, $nb_iter0, $nb_iter1,'${outPathZ}Semidiscrete_',$Nmax,$Npart,Zfile='${outPathZ}originalZnu_ZwC1.0_sig25.0_semidiscrete_plus0.05.npz',maxV=$maxV,optMethod='$optMethod');quit()" >> "${outPathZ}Semidiscrete_C1.0.txt"
    echo $(date) >> "${outPathZ}Semidiscrete_C1.0.txt"
done

