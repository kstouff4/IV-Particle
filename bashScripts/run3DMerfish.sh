#!/bin/bash

cd /cis/home/kstouff4/Documents/MeshRegistration/Scripts-KMS/approxCode

nb_iter0=4
nb_iter1=15
Nmax=5000.0
Npart=1000.0
optMethod="LBFGS"
sigma=0.025

atlasImage="/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenMerfish/XnuX/"
targetImage="/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenMerfish/ZApprox-XComb_sig25.0/"
outPath="/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenMerfish"
outPathX=$outPath'/XSplits/'
outPathZ=$outPath"/ZApprox-XComb_sig${sigma}/"
zname="originalZnu_ZwC1.0_sig25.0_semidiscrete_plus0.05"
maxV=702

fils=$(find $outPathX | grep XnuX._3 | grep npz )
for f in ${fils[*]}; do
    pref="$(basename -- $f)"
    pref2=$(echo $pref | tr X Z)
    pref3=${pref2%.npz}
    outPref="${outPathZ}"
    echo "x file is $f"
    echo $(date) >> "$outPathZ${pref3}_${Nmax}_$Npart.txt"
    echo "z file is ${pref/XnuX/$zname}"
    if [[ -f "$outPathZ${pref3}__optimalZnu_ZAllwC1.0_sig0.025_Nmax5000.0_Npart1000.0.npz" ]]; then
        echo "already did"
        continue
    fi
    python3 -c "import estimateSubsampleByLabelScratchTestExperiments as ess; ess.project3D('$f',$sigma, $nb_iter0, $nb_iter1,'$outPathZ${pref3}_',$Nmax,$Npart,Zfile='$targetImage${pref/XnuX/$zname}',maxV=$maxV,optMethod='$optMethod',C=1.0);quit()" >> "$outPathZ${pref3}_${Nmax}_$Npart.txt"
    echo $(date) >> "$outPathZ${pref3}_${Nmax}_$Npart.txt"
done

