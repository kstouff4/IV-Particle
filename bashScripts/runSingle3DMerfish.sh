#!/bin/bash

cd /cis/home/kstouff4/Documents/MeshRegistration/Scripts-KMS/approxCode

nb_iter0=3
nb_iter1=10
Nmax=1500.0
Npart=2000.0
optMethod="LBFGS"
sigma=0.025
num='202'

targetImage="/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenMerfish/ZApprox-XComb_sig25.0/"
outPath="/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenMerfish"
outPathX=$outPath'/XnuX/'
outPathZ=$outPath"/ZApprox-XComb_sig${sigma}_Single/"
zname="originalZnu_ZwC1.0_sig25.0_semidiscrete_plus0.05"
maxV=702
mkdir $outPathZ

fils=$(find $outPathX | grep XnuX | grep $num | grep npz )
fils=$(echo $fils | xargs -n1 | sort | xargs) # sort so that do parts 0-3 of same slice consecutively 
for f in ${fils[*]}; do
    pref="$(basename -- $f)"
    pref2=$(echo $pref | tr X Z)
    pref3=${pref2%.npz}
    outPref="${outPathZ}"
    echo "x file is $f"
    echo $(date) >> "$outPathZ${pref3}_${Nmax}_${Npart}.txt"
    echo "z file is ${pref/XnuX/$zname}"
    if [[ -f "$outPathZ${pref3}__optimalZnu_ZAllwC1.0_sig0.025_Nmax${Nmax}_Npart${Npart}.npz" ]]; then
        echo "already did"
        continue
    fi
    python3 -c "import approximateFine as ess; ess.project3D('$f',$sigma, $nb_iter0, $nb_iter1,'$outPathZ${pref3}_',$Nmax,$Npart,Zfile='$targetImage${pref/XnuX/$zname}',maxV=$maxV,optMethod='$optMethod',C=1.0);quit()" >> "$outPathZ${pref3}_${Nmax}_${Npart}.txt"
    echo $(date) >> "$outPathZ${pref3}_${Nmax}_${Npart}.txt"
done

