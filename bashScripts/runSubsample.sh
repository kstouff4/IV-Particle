#!/bin/bash

atlasImage="/cis/home/kstouff4/Documents/MeshRegistration/TestImages/Allen_10_anno_16bit_ap.img"
ax=2
indS=920
indF=950
outPath="/cis/home/kstouff4/Documents/MeshRegistration/Particles/${indS}-${indF}SliceAllenWindow"
#outDir="/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um/"
#outPath="${outDir}Sub"
#outPath="${outDir}Sub_0-100_"

eps=0.15 # within 4 std of mean 
sig=0.05
nb_iter0=3
nb_iter1=15

# Yongsoo
atlasImage="/cis/home/kstouff4/Documents/MeshRegistration/TestImages/Yongsoo/KimLabDevCCFv001_Annotations_ASL_Oriented_10um.nii"
ax=0

partFile="/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um/Sub_1200-1300_XnuX.npz"
python3 -c "import estimateSubsample as ess; ess.splitParticles('$partFile',2,ax0=True,ax1=True,ax2=False);quit()"

partFile="/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um/Sub_1000-1100_XnuX.npz"
python3 -c "import estimateSubsample as ess; ess.splitParticles('$partFile',2,ax0=True,ax1=True,ax2=False);quit()"

partFile="/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um/Sub_1100-1200_XnuX.npz"
python3 -c "import estimateSubsample as ess; ess.splitParticles('$partFile',2,ax0=True,ax1=True,ax2=False);quit()"


#python3 -c "import estimateSubsample as ess; ess.makeAllXandZ('$atlasImage','$outPath', thickness=100, res=0.01,sig=0.1,C=-1,flip=True); quit()"

#python3 -c "import estimateSubsample as ess; ess.compileX('$outPath');quit()"

outPath="/cis/home/kstouff4/Documents/MeshRegistration/Particles/SubsampledAllen10umAtlas/Sub"
atlasImage="/cis/home/kstouff4/Documents/MeshRegistration/TestImages/Allen_10_anno_16bit_ap_subsamp2.img"
#python3 -c "import estimateSubsample as ess; ess.makeAllXandZ('$atlasImage','$outPath', thickness=50, res = 0.02, sig=0.1, C=-1,flip=False); quit()"

#python3 -c "import estimateSubsample as ess; ess.compileX('$outPath');quit()"

#python3 -c "import estimateSubsampleByLabel as ess; ess.project3D('${outPath}XnuX.npz',$eps, $sig, $nb_iter0, $nb_iter1,'$outPath','${outPath}ZnuZ.npz');quit()"

outPath="${outDir}Sub_100-200_"
#python3 -c "import estimateSubsampleByLabel as ess; ess.project3D('${outPath}XnuX.npz',$eps, $sig, $nb_iter0, $nb_iter1,'$outPath','${outPath}ZnuZ.npz');quit()"


outPath="${outDir}Sub_200-300_"
#python3 -c "import estimateSubsampleByLabel as ess; ess.project3D('${outPath}XnuX.npz',$eps, $sig, $nb_iter0, $nb_iter1,'$outPath','${outPath}ZnuZ.npz');quit()"
#python3 -c "import estimateSubsampleSparse as ess; X, nu_X, Z, nu_Z, indsToKeep = ess.getXZ('${outPath}XnuX.npz','${outPath}ZnuZ.npz'); ess.project3D(X,nu_X,$eps, $sig, $nb_iter0, $nb_iter1,'$outPath',Z,nu_Z,oneHot=False);quit()"

outPath="${outDir}Sub_300-400_"
#python3 -c "import estimateSubsampleByLabel as ess; ess.project3D('${outPath}XnuX.npz',$eps, $sig, $nb_iter0, $nb_iter1,'$outPath','${outPath}ZnuZ.npz');quit()"
#python3 -c "import estimateSubsampleSparse as ess; X, nu_X, Z, nu_Z, indsToKeep, d = ess.getXZInt('${outPath}XnuX.npz','${outPath}ZnuZ.npz'); ess.project3D(X,nu_X,$eps, $sig, $nb_iter0, $nb_iter1,'$outPath',Z,nu_Z);quit()"

outPath="${outDir}Sub_400-500_"
#python3 -c "import estimateSubsampleByLabel as ess; ess.project3D('${outPath}XnuX.npz',$eps, $sig, $nb_iter0, $nb_iter1,'$outPath','${outPath}ZnuZ.npz');quit()"
#python3 -c "import estimateSubsampleSparse as ess; X, nu_X, Z, nu_Z, indsToKeep, d = ess.getXZInt('${outPath}XnuX.npz','${outPath}ZnuZ.npz'); ess.project3D(X,nu_X,$eps, $sig, $nb_iter0, $nb_iter1,'$outPath',Z,nu_Z);quit()"

outPath="${outDir}Sub_500-600_"
#python3 -c "import estimateSubsampleByLabel as ess; ess.project3D('${outPath}XnuX.npz',$eps, $sig, $nb_iter0, $nb_iter1,'$outPath','${outPath}ZnuZ.npz');quit()"
#python3 -c "import estimateSubsampleSparse as ess; X, nu_X, Z, nu_Z, indsToKeep, d = ess.getXZInt('${outPath}XnuX.npz','${outPath}ZnuZ.npz'); ess.project3D(X,nu_X,$eps, $sig, $nb_iter0, $nb_iter1,'$outPath',Z,nu_Z);quit()"

outPath="${outDir}Sub_600-700_"
#python3 -c "import estimateSubsampleByLabel as ess; ess.project3D('${outPath}XnuX.npz',$eps, $sig, $nb_iter0, $nb_iter1,'$outPath','${outPath}ZnuZ.npz');quit()"

outPath="${outDir}Sub_700-800_"
#python3 -c "import estimateSubsampleByLabel as ess; ess.project3D('${outPath}XnuX.npz',$eps, $sig, $nb_iter0, $nb_iter1,'$outPath','${outPath}ZnuZ.npz');quit()"

outPath="${outDir}Sub_800-900_"
#python3 -c "import estimateSubsampleByLabel as ess; ess.project3D('${outPath}XnuX.npz',$eps, $sig, $nb_iter0, $nb_iter1,'$outPath','${outPath}ZnuZ.npz');quit()"

outPath="${outDir}Sub_900-1000_"
#python3 -c "import estimateSubsampleByLabel as ess; ess.project3D('${outPath}XnuX.npz',$eps, $sig, $nb_iter0, $nb_iter1,'$outPath','${outPath}ZnuZ.npz');quit()"

outPath="${outDir}Sub_1000-1100_"
#python3 -c "import estimateSubsampleByLabel as ess; ess.project3D('${outPath}XnuX.npz',$eps, $sig, $nb_iter0, $nb_iter1,'$outPath','${outPath}ZnuZ.npz');quit()"

outPath="${outDir}Sub_1100-1200_"
#python3 -c "import estimateSubsampleByLabel as ess; ess.project3D('${outPath}XnuX.npz',$eps, $sig, $nb_iter0, $nb_iter1,'$outPath','${outPath}ZnuZ.npz');quit()"

outPath="${outDir}Sub_1200-1300_"
#python3 -c "import estimateSubsampleByLabel as ess; ess.project3D('${outPath}XnuX.npz',$eps, $sig, $nb_iter0, $nb_iter1,'$outPath','${outPath}ZnuZ.npz');quit()"

#python3 -c "import estimateSubsample as ess; ess.estimateSubSampleImg('$atlasImage','$outPath',ax=$ax,indS=$indS,indF=$indF,res=0.01,sig=0.01,C=1); quit()"

#python3 -c "import estimateSubsample as ess; ess.estimateSubSampleImg('$atlasImage','$outPath',ax=$ax,indS=$indS,indF=$indF,res=0.01,sig=1,C=1); quit()"


indS=820
indF=840
outPath="/cis/home/kstouff4/Documents/MeshRegistration/Particles/${indS}-${indF}SliceAllenWindow"

#python3 -c "import estimateSubsample as ess; ess.estimateSubSampleImg('$atlasImage','$outPath',ax=$ax,indS=$indS,indF=$indF,res=0.01,sig=0.1,C=1); quit()"

indS=620
indF=640
outPath="/cis/home/kstouff4/Documents/MeshRegistration/Particles/${indS}-${indF}SliceAllenWindow"

#python3 -c "import estimateSubsample as ess; ess.estimateSubSampleImg('$atlasImage','$outPath',ax=$ax,indS=$indS,indF=$indF,res=0.01,sig=0.1,C=1); quit()"

indS=920
indF=960
outPath="/cis/home/kstouff4/Documents/MeshRegistration/Particles/${indS}-${indF}SliceAllenWindow"

#python3 -c "import estimateSubsample as ess; ess.estimateSubSampleImg('$atlasImage','$outPath',ax=$ax,indS=$indS,indF=$indF,res=0.01,sig=0.1,C=1); quit()"


