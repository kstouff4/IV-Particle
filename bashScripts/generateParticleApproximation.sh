#!/bin/bash

cd /cis/home/kstouff4/Documents/MeshRegistration/Scripts-KMS/approxCode

# Examples:
# YS Kim Atlas: ./generateParticleApproximation.sh -s True -z 2 -d kim -w 0.05 -o /cis/home/kstouff4/Documents/MeshRegistration/Particles/KimAtlas10um -r True
# Allen: ./generateParticleApproximation.sh -s True -z 2 -d allen -w 0.1 -o /cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um -r True
# Allen3d: ./generateParticleApproximation.sh -s False -z 2 -d allen3d -w 0.025 -o /cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenMerfish -r True

while getopts s:z:d:w:o:r: flag
do
    case "${flag}" in
        s) slabs=${OPTARG};; # divide dataset into slabs along zAxis 
        z) zAxis=${OPTARG};; # axis index (0-based) along which slabs are made (anterior to posterior)
        d) dataset=${OPTARG};; # indicates set of parameters or file name with text file with parameters
        w) sigma=${OPTARG};; # sigma indicating resolution of resampling (in mm units)
        o) outPath=${OPTARG};; # output path to which to write (will make subdirectory of XSplits
        r) resampleOnly=${OPTARG};; # set to False if only doing approximation and not splitting
    esac
done

mkdir $outPath
mkdir $outPath'/XSplits/'

outPathX=$outPath'/XSplits/'
outPathZ=$outPath"/ZApprox-XComb_sig${sigma}/"
mkdir $outPathZ

if [[ $dataset == "allen" ]]; then
    thick=100 # number of slices to compile into slab
    res=0.01 # mm resolution of original image
    atlasImage="/cis/home/kstouff4/Documents/MeshRegistration/TestImages/Allen_10_anno_16bit_ap.img"
    flip="False"
    list="False" # list is true if have a list of files to turn to particles
    maxV=673
elif [[ $dataset == "kim" ]]; then
    thick=100
    res=0.01
    atlasImage="/cis/home/kstouff4/Documents/MeshRegistration/TestImages/Yongsoo/KimLabDevCCFv001_Annotations_ASL_Oriented_10um.nii"
    flip="True"
    list="False"
    maxV=352

elif [[ $dataset == "allen3d" ]]; then
    list="True"
    atlasImage="/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenMerfish/XnuX/"
    targetImage="/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenMerfish/ZApprox-XComb_sig25.0/"
    #outPathX=$outPath'/XnuX/'
    outPathZ=$outPath"/ZApprox-XComb_sig${sigma}/"
    zname="originalZnu_ZwC1.0_sig25.0_semidiscrete_plus0.05"
    maxV=702

else
    # read in text file with parameters
    echo "file path to atlas"
    read atlasImage
    echo "is list? type True or False"
    read list
    echo "maxV"
    read maxV
fi

echo "computing with the following"
echo $atlasImage
echo $list
echo $maxV
    
# Divide Dataset into Slabs (~1 mm thick) and split into quadrants 
if [[ $resampleOnly == "False" ]]; then
    if [[ $slabs == "True" ]]; then
        python3 -c "import estimateSubsample as ess; xfiles = ess.makeAllXandZ('$atlasImage','$outPathX', thickness=$thick, res=$res,sig=$sigma,C=-1,flip=$flip); ess.splitParticlesList(xfiles,2,ax0=True,ax1=True,ax2=False); quit()"
    elif [[ $list == "True" ]]; then
        python3 -c "import estimateSubsample as ess; xfiles = ess.getFiles('$atlasImage','X.npz');ess.rescale(xfiles,s=1e-3); ess.splitParticlesList(xfiles,2,ax0=True,ax1=True,ax2=False); zfiles=ess.getFiles('$targetImage','semidiscrete_plus0.05.npz'); ess.rescale(zfiles,s=1e-3);ess.splitParticlesList(zfiles,2,ax0=True,ax1=True,ax2=False);quit()"
    else
        python3 -c "import estimateSubsample as ess; ess.splitParticles('$atlasImage',2,ax0=True,ax1=True,ax2=False); quit()"
    fi
fi

# Approximate Particles for each quadrant of each slab

nb_iter0=4
nb_iter1=15
Nmax=5000.0
Npart=1000.0
optMethod="LBFGS"

# look for all of the Quadrants in folder based on X's and generate all
# This is for subsampling as well as approximating
fils=$(find $outPathX | grep XnuX._3 | grep npz )
for f in ${fils[*]}; do
    pref="$(basename -- $f)"
    pref2=$(echo $pref | tr X Z)
    pref3=${pref2%.npz}
    outPref="${outPathZ}"
    echo "x file is $f"
    echo $(date) >> "$outPathZ${pref3}_${Nmax}_$Npart.txt"
    if [[ $dataset == "allen3d" ]]; then
        echo "z file is ${pref/XnuX/$zname}"
        python3 -c "import estimateSubsampleByLabelScratchTestExperiments as ess; ess.project3D('$f',$sigma, $nb_iter0, $nb_iter1,'$outPathZ${pref3}_',$Nmax,$Npart,Zfile='$targetImage${pref/XnuX/$zname}',maxV=$maxV,optMethod='$optMethod',C=1.0);quit()" >> "$outPathZ${pref3}_${Nmax}_$Npart.txt"
    else
        python3 -c "import estimateSubsampleByLabelScratchTestExperiments as ess; ess.project3D('$f',$sigma, $nb_iter0, $nb_iter1,'$outPathZ${pref3}_',$Nmax,$Npart,maxV=$maxV,optMethod='$optMethod');quit()" >> "$outPathZ${pref3}_${Nmax}_$Npart.txt"
    fi
    echo $(date) >> "$outPathZ${pref3}_${Nmax}_$Npart.txt"
done

# Stitch quadrants 
margin=2.0
nb_iter0=3
nb_iter1=7
fils=$(find $outPathX | grep X\.npz) # gets all of slabs
for f in ${fils[*]}; do
    pref1="$(basename -- $f)"
    pref=${pref1%_XnuX.npz} # should be prefix to look for
    echo "prefix is $pref"
    #python3 -c "import smootheBoundaries as sb; sb.stitchAllQuadrants('$outPathX','$outPathZ','$pref',$sigma,$margin,'$outPathZ',$nb_iter0,$nb_iter1,$Nmax,$Npart,$maxV); quit()" >> ${outPathZ}_$pref.txt
done

#python3 -c "import smootheBoundaries as sb; sb.stitchAllSlabs(['$outPathZ'],['$outPathZ'],'_0123_',$sigma,$margin,'$outPathZ',$nb_iter0,$nb_iter1,$Nmax,$Npart,$maxV);quit()"

#python3 -c "import smootheBoundaries as sb; sb.stitchTwoSlabs(['$outPathZ'],['$outPathZ'],'_0123_',$sigma,$margin,'$outPathZ',$nb_iter0,$nb_iter1,$Nmax,$Npart,$maxV);quit()"

