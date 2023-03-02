#!/bin/bash
cd /cis/home/kstouff4/Documents/MeshRegistration/Scripts-KMS/approxCode

sigma=0.05
outPathZ="/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenMerfish/InitialZQuadrants/"
mkdir $outPathZ
delta=0.05

inpathX='/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenMerfish/XSplits/'

fils=$(find $inpathX | grep XnuX._ | grep npz | grep 20)
for f in ${fils[*]}; do
    pref="$(basename -- $f)"
    pref2=$(echo $pref | tr X Z)
    pref3=${pref2%.npz}
    outPref="${outPathZ}"
    python3 -c "fs='$f'; pref=fs.split('/')[-2]; p=pref.split('_')[1]; print(p); r=p[-3:]; r=int(r); z = (r-201)*100;import subsampleFunctions as sf; sf.makeSubsampleStratified('$f','X','nu_X', $sigma, '${savename}_' + str(r), xtype='discrete',ztype='semi-discrete',overhead=$delta,maxV=702,C=1.0,dim=2,z=z); quit()"    
done



