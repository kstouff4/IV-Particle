#!/bin/bash

cd /cis/home/kstouff4/Documents/MeshRegistration/Scripts-KMS/approxCode

step=2
substep=2

nb_iter0=0
nb_iter1=50
Nmax=1500.0
Npart=2000.0
optMethod="LBFGS"
s0=0.1
s1=0.5
s2=1.0


if [[ $step == 1 ]]; then
    atlasImage="/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenMerfish/ZnuZ_Aligned/top20MI/sig0.1/All_ZnuZ_sig0.05.npz"
    targetImage="/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenMerfish/ZnuZ_Aligned/top20MI/sig0.1/sig0.1__originalZnu_ZwC1.2_sig0.1_semidiscrete.npz"
    outPath="/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenMerfish/ZnuZ_Aligned/top20MI/sig0.1/"
    maxV=20
    mkdir $outPath

    #python3 -c "import subsampleFunctions as sf; sf.makeSubsampleStratified('$atlasImage','Z','nu_Z',$s0,'${outPath}All_ZnuZ_sig0.05',xtype='semi-discrete',ztype='uniform',overhead=0.05,maxV=$maxV,C=1.0,dim=3,saveX=False,zeroBased=True,alpha=0.5);quit()"
    if [[ $substep == 1 ]]; then
        python3 -c "import subsampleFunctions as sf; sf.makeSubsample('$atlasImage',$s0,'${outPath}sig0.1_',xtype='semi-discrete',ztype='semi-discrete',overhead=0.05,maxV=$maxV,C=1.2);quit()"
    else

        echo $(date) > "${outPath}All_ZnuZ_alpha.txt"

        python3 -c "import approximateIntermediateMultiScaleNewLoss as ess; ess.project3D('$atlasImage',[$s0], $nb_iter0, $nb_iter1,'${outPath}All_ZnuZ_',$Nmax,$Npart,Zfile='$targetImage',maxV=$maxV,optMethod='$optMethod',C=1.2);quit()" >> "${outPath}All_ZnuZ_alpha.txt"

        echo $(date) >> "${outPath}All_ZnuZ_alpha.txt"
    fi
elif [[ $step == 2 ]]; then

    atlasImage="/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenMerfish/ZnuZ_Aligned/top20MI/sig0.1/All_ZnuZ__optimalZnu_ZAllwC1.2_sig[0.1]_Nmax1500.0_Npart2000.0.npz"
    outPath="/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenMerfish/ZnuZ_Aligned/top20MI/sig0.2/"
    targetImage="/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenMerfish/ZnuZ_Aligned/top20MI/sig0.2/sig0.2__originalZnu_ZwC1.2_sig0.2_semidiscrete.npz"

    maxV=20
    mkdir $outPath
    s0=0.2

    if [[ $substep == 1 ]]; then
        # blurring to 0.2, and then 0.4
        python3 -c "import subsampleFunctions as sf; sf.makeSubsample('$atlasImage',$s0,'${outPath}sig0.2_',xtype='semi-discrete',ztype='semi-discrete',overhead=0.05,maxV=$maxV,C=1.2);quit()"
    else
    
        echo $(date) > "${outPath}All_ZnuZ_alpha.txt"

        python3 -c "import approximateIntermediateMultiScaleNewLoss as ess; ess.project3D('$atlasImage',[$s0], $nb_iter0, $nb_iter1,'${outPath}All_ZnuZ_',$Nmax,$Npart,Zfile='$targetImage',maxV=$maxV,optMethod='$optMethod',C=1.2);quit()" >> "${outPath}All_ZnuZ_alpha.txt"

        echo $(date) >> "${outPath}All_ZnuZ_alpha.txt"
    fi
fi

