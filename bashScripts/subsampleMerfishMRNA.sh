#!/bin/bash
cd /cis/home/kstouff4/Documents/MeshRegistration/Scripts-KMS/approxCode
sigma=25.0
savename="/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenMerfish/"
mkdir $savename
delta=0.05

fils=$(find /cis/home/kstouff4/Documents/SpatialTranscriptomics/Mouse/Mouse1_20220506/zipfiles1/ | grep detected_transcripts.npz | grep -v hk | grep -v HK)
for f in ${fils[*]}; do
    echo $f
    python3 -c "fs='$f'; pref=fs.split('/')[-2]; p=pref.split('_')[1]; print(p); r=p[-3:]; r=int(r); z = (r-201)*100;import subsampleFunctions as sf; sf.makeSubsampleStratified('$f','coordsTot','geneInd', $sigma, '${savename}_' + str(r), xtype='discrete',ztype='semi-discrete',overhead=$delta,maxV=702,C=1.0,dim=2,z=z); quit()"
done

Xfile="/cis/home/kstouff4/Documents/SpatialTranscriptomics/Mouse/Mouse1_20220506/zipfiles1/202202170851_60988201_VMSC01001/detected_transcripts.npz"
savename="${savename}201_"
delta=0.05

#python3 -c "import subsampleFunctions as sf; sf.makeSubsampleStratified('$Xfile','coordsTot','geneInd',$sigma,'$savename',xtype='discrete',ztype='semi-discrete',overhead=$delta,maxV=702,C=1.0,dim=2,z=100); quit()"

Xfile="/cis/home/kstouff4/Documents/SpatialTranscriptomics/Mouse/Mouse1_20220506/zipfiles1/202202170855_60988202_VMSC01601/detected_transcripts.npz"
savename="${savename}202_"

#python3 -c "import subsampleFunctions as sf; sf.makeSubsampleStratified('$Xfile','coordsTot','geneInd',$sigma,'$savename',xtype='discrete',ztype='semi-discrete',overhead=$delta,maxV=702,C=1.0,dim=2,z=200); quit()"

Xfile="/cis/home/kstouff4/Documents/SpatialTranscriptomics/Mouse/Mouse1_20220506/zipfiles1/202202170915_60988203_VMSC00401/detected_transcripts.npz"
savename="${savename}203_"

#python3 -c "import subsampleFunctions as sf; sf.makeSubsampleStratified('$Xfile','coordsTot','geneInd',$sigma,'$savename',xtype='discrete',ztype='semi-discrete',overhead=$delta,maxV=702,C=1.0,dim=2,z=300); quit()"

Xfile="/cis/home/kstouff4/Documents/SpatialTranscriptomics/Mouse/Mouse1_20220506/zipfiles1/202202180951_60988204_VMSC01001/detected_transcripts.npz"
savename="${savename}204_"

#python3 -c "import subsampleFunctions as sf; sf.makeSubsampleStratified('$Xfile','coordsTot','geneInd',$sigma,'$savename',xtype='discrete',ztype='semi-discrete',overhead=$delta,maxV=702,C=1.0,dim=2,z=400); quit()"

