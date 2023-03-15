#!/bin/bash

cd /cis/home/kstouff4/Documents/MeshRegistration/Scripts-KMS/approxCode

nb_iter0=7
nb_iter1=20
Nmax=1500.0
Npart=2000.0
optMethod="LBFGS"
sigma=0.05

atlasImage="/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenMerfish/XnuX/"
targetImage="/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenMerfish/InitialZTotalUniform/"
outPath="/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenMerfish"
outPathX=$outPath'/XSplits/'
outPathZ=$outPath"/ZApprox_sig${sigma}Uniform/" #"/ZApprox-XComb_sig${sigma}/"
zname="originalZnu_ZwC1.0_sig0.05_uniform"
maxV=702
mkdir $outPathZ

fils=$(find $targetImage | grep XnuX | grep npz)
fils=$(echo $fils | xargs -n1 | sort | xargs) # sort so that do parts 0-3 of same slice consecutively 
for f in ${fils[*]}; do
    pref="$(basename -- $f)"
    pref2=$(echo $pref | tr X Z)
    pref3=${pref2%.npz}
    outPref="${outPathZ}"
    echo "x file is $f"
    if [[ -f "$outPathZ${pref3}__optimalZnu_ZAllwC1.0_sig0.05_Nmax${Nmax}_Npart${Npart}.npz" ]]; then
        echo "already did"
        continue
    fi
    echo $(date) >> "$outPathZ${pref3}_${Nmax}_${Npart}.txt"
    echo "z file is ${pref/XnuX/$zname}"
    python3 -c "import approximateFine as ess; ess.project3D('$f',$sigma, $nb_iter0, $nb_iter1,'$outPathZ${pref3}_',$Nmax,$Npart,Zfile='$targetImage${pref/XnuX/$zname}',maxV=$maxV,optMethod='$optMethod',C=1.0);quit()" >> "$outPathZ${pref3}_${Nmax}_${Npart}.txt"
    echo $(date) >> "$outPathZ${pref3}_${Nmax}_${Npart}.txt"
done

