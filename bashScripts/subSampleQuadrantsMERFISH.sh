#!/bin/bash
cd /cis/home/kstouff4/Documents/MeshRegistration/Scripts-KMS/approxCode

sigma=0.05
outPathZ="/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenMerfish/InitialZTotalUniform/"
mkdir $outPathZ
delta=0.05

inpathX='/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenMerfish/XnuX/'

fils=$(find $inpathX | grep XnuX | grep npz) # _###_XnuX._i.npz
for f in ${fils[*]}; do
    echo $f
    pref="$(basename -- $f)"
    pref2=$(echo $pref | tr X Z)
    pref3=${pref2%.npz}
    outPref="${outPathZ}"
    python3 -c "fs='$f'; pref=fs.split('/')[-1]; p=pref[1:4]; print(p); r=int(p); p = pref.replace('.npz',''); i=p[-1]; z = (r-201)*100;import subsampleFunctions as sf; sf.makeSubsampleStratified('$f','X','nu_X', $sigma, '${outPathZ}_' + str(r) + '_' + str(i), xtype='discrete',ztype='semi-discrete',overhead=$delta,maxV=702,C=1.0,dim=2,z=z); quit()"    
done



