#!/bin/bash

inpathX='/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um/'
inpathZ='/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um/Experiments/'
outpath='/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um/Experiments/Stitched/'
outpath='/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um/ZApprox-XComb_sig0.1/'
inpathX='/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um/XSplits/'
inpathZ=$outpath
mkdir $outpath

sigma=0.1
margin=2.0
nb_iter0=3
nb_iter1=7
NmaxI=5000.0
NpartI=1000.0
maxVal=673
#maxVal=352

nb_iter0=3
nb_iter1=10
Nmax=5000.0
Npart=1000.0
optMethod="LBFGS"

# look for all of the Quadrants in folder based on X's and generate all
fils=$(find $inpathX | grep XnuX._ | grep npz | grep Sub_0)
for f in ${fils[*]}; do
    pref="$(basename -- $f)"
    pref2=$(echo $pref | tr X Z)
    pref3=${pref2%.npz}
    outPref="${outPath}"
    #echo $(date) >> "$outPath${pref3}_${Nmax}_$Npart.txt"
    #python3 -c "import estimateSubsampleByLabelScratchTestExperiments as ess; ess.project3D('$f',$sigma, $nb_iter0, $nb_iter1,'$outPath${pref3}_',$Nmax,$Npart,maxV=$maxVal,optMethod='$optMethod');quit()" >> "$outPath${pref3}_${Nmax}_$Npart.txt"
    #echo $(date) >> "$outPath${pref3}_${Nmax}_$Npart.txt"
done

pref='_0400-0500'
#python3 -c "import smootheBoundaries as sb; sb.stitchAllQuadrants('$inpathX','$inpathZ','$pref',$sigma,$margin,'$outpath',$nb_iter0,$nb_iter1,$NmaxI,$NpartI,$maxVal); quit()" >> $outpath${pref}_log3.txt

pref='_0500-0600'
#python3 -c "import smootheBoundaries as sb; sb.stitchAllQuadrants('$inpathX','$inpathZ','$pref',$sigma,$margin,'$outpath',$nb_iter0,$nb_iter1,$NmaxI,$NpartI,$maxVal); quit()" >> $outpath${pref}_log3.txt

pref='_0600-0700'
#python3 -c "import smootheBoundaries as sb; sb.stitchAllQuadrants('$inpathX','$inpathZ','$pref',$sigma,$margin,'$outpath',$nb_iter0,$nb_iter1,$NmaxI,$NpartI,$maxVal); quit()" >> $outpath${pref}_log3.txt

pref='_0700-0800'
#python3 -c "import smootheBoundaries as sb; sb.stitchAllQuadrants('$inpathX','$inpathZ','$pref',$sigma,$margin,'$outpath',$nb_iter0,$nb_iter1,$NmaxI,$NpartI,$maxVal); quit()" >> $outpath${pref}_log3.txt

pref='_0800-0900'
#python3 -c "import smootheBoundaries as sb; sb.stitchAllQuadrants('$inpathX','$inpathZ','$pref',$sigma,$margin,'$outpath',$nb_iter0,$nb_iter1,$NmaxI,$NpartI,$maxVal); quit()" >> $outpath${pref}_log3.txt

pref='_0600-0700'
#python3 -c "import smootheBoundaries as sb; sb.stitchAllQuadrants('$inpathX','$inpathZ','$pref',$sigma,$margin,'$outpath',$nb_iter0,$nb_iter1,$NmaxI,$NpartI,$maxVal); quit()" >> $outpath${pref}_log3.txt

pref='_0700-0800'
#python3 -c "import smootheBoundaries as sb; sb.stitchAllQuadrants('$inpathX','$inpathZ','$pref',$sigma,$margin,'$outpath',$nb_iter0,$nb_iter1,$NmaxI,$NpartI,$maxVal); quit()" >> $outpath${pref}_log3.txt

pref='_0900-1000'
python3 -c "import smootheBoundaries as sb; sb.stitchAllQuadrants('$inpathX','$inpathZ','$pref',$sigma,$margin,'$outpath',$nb_iter0,$nb_iter1,$NmaxI,$NpartI,$maxVal); quit()" >> $outpath${pref}_log3.txt

fX01='/cis/home/kstouff4/Documents/MeshRegistration/Particles/KimAtlas10um/ZApprox-XComb/Sub__1100-1200_01_XnuX.npz'
fX23='/cis/home/kstouff4/Documents/MeshRegistration/Particles/KimAtlas10um/ZApprox-XComb/Sub__1100-1200_23_XnuX.npz'
fZ01='/cis/home/kstouff4/Documents/MeshRegistration/Particles/KimAtlas10um/ZApprox-XComb/Sub__1100-1200_01_optimalZnu_ZAllwC0.2_sig0.05_Nmax5000.0_Npart0000.0.npz'
fZ23='/cis/home/kstouff4/Documents/MeshRegistration/Particles/KimAtlas10um/ZApprox-XComb/Sub__1100-1200_23_optimalZnu_ZAllwC0.2_sig0.05_Nmax5000.0_Npart0000.0.npz'
#python3 -c "import smootheBoundaries as sb; sb.stitchQuadrants('$fX01','$fX23','$fZ01','$fZ23','$outpath' + 'Sub_' + '_1100-1200' + '_0123',$sigma,$nb_iter0,$nb_iter1,margin=$margin, Nmax=$NmaxI, Npart=$NpartI,maxV=$maxVal,optMethod='LBFGS');quit()"

pref='_900-1000'
#python3 -c "import smootheBoundaries as sb; sb.stitchAllQuadrants('$inpathX','$inpathZ','$pref',$sigma,$margin,'$outpath',$nb_iter0,$nb_iter1,$NmaxI,$NpartI,$maxVal); quit()" >> $outpath${pref}_log3.txt

pref='_0123_'
#python3 -c "import smootheBoundaries as sb; sb.stitchTwoSlabs('$inpathZ','$inpathZ','$pref',$sigma,$margin,'$outpath',$nb_iter0,$nb_iter1,$NmaxI,$NpartI,$maxVal); quit()" > $outpath${pref}_logNew.txt

