#!/bin/bash

cd /cis/home/kstouff4/Documents/MeshRegistration/Scripts-KMS/approxCode

nb_iter0=7
nb_iter1=20
Nmax=1500.0
Npart=2000.0
optMethod="LBFGS"
sigma=0.05

atlasImage="/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenMerfish/XnuX_Aligned/top20MI/"
targetImage="/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenMerfish/InitialZTotalUniform/"
outPath="/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenMerfish/ZnuZ_Aligned/"
mkdir $outPath
outPathX=$outPath'/XSplits/'
outPathZ=$outPath"/top20MI/" #"/ZApprox-XComb_sig${sigma}/"
zname="originalZnu_ZwC1.0_sig0.05_uniform"
maxV=20
mkdir $outPathZ

fils=$(find $targetImage | grep XnuX | grep npz)
fils=$(echo $fils | xargs -n1 | sort | xargs) # sort so that do parts 0-3 of same slice consecutively 
fils=$(find $atlasImage | grep XnuX | grep npz )
for f in ${fils[*]}; do
    pref="$(basename -- $f)"
    pref2=$(echo $pref | tr X Z)
    pref3=${pref2%.npz.npz}
    outPref="${outPathZ}"
    echo "x file is $f"
    if [[ -f "$outPathZ${pref3}__optimalZnu_ZAllwC8.0_sig0.05_Nmax${Nmax}_Npart${Npart}.npz" ]]; then
        echo "already did"
        continue
    fi
    #python3 -c "import subsampleFunctions as sf; sf.makeSubsampleStratified('$f','X','nu_X',$sigma,'${outPathZ}${pref3}_',xtype='discrete',ztype='uniform',overhead=0.1,maxV=$maxV,Co=1.0,dim=2,saveX=False,zeroBased=True,alpha=0.5);quit()"
    
    echo $(date) >> "$outPathZ${pref3}_${Nmax}_${Npart}.txt"
    python3 -c "import approximateIntermediateMultiScale as ess; ess.project3D('$f',[$sigma], $nb_iter0, $nb_iter1,'${outPathZ}${pref3}_',$Nmax,$Npart,Zfile='${outPathZ}${pref3}__originalZnu_ZwC8.0_sig0.05_uniform.npz',maxV=$maxV,optMethod='$optMethod',C=8.0);quit()" >> "$outPathZ${pref3}_${Nmax}_${Npart}.txt"

    #python3 -c "import approximateFine as ess; ess.project3D('$f',$sigma, $nb_iter0, $nb_iter1,'$outPathZ${pref3}_',$Nmax,$Npart,Zfile='$targetImage${pref/XnuX/$zname}',maxV=$maxV,optMethod='$optMethod',C=1.0);quit()" >> "$outPathZ${pref3}_${Nmax}_${Npart}.txt"
    echo $(date) >> "$outPathZ${pref3}_${Nmax}_${Npart}.txt"
done

