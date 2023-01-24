#!/bin/bash

# run with steps: ./runMI.sh 1 allen or ./runMI.sh 2 merfish or ./runMI.sh 3 (mouse steps)

k=4
mSize=6
cSize=50.0

if [[ $2 == 'allen' ]]; then
    fp='/cis/home/kstouff4/Documents/SpatialTranscriptomics/Mouse/Mouse1_20220506/zipfiles1/'
    fils=$(find $fp | grep detected_transcripts.csv)

    if [[ $1 == 1 ]]; then
        for f in ${fils[*]}; do
            python3 -c "import mrnaMI as mm; mm.singleMI('$f',$cSize,$mSize,$k); quit()"
        done
    elif [[ $1 == 2 ]]; then
        fp='/cis/home/kstouff4/Documents/SpatialTranscriptomics/Mouse/Mouse1_20220506/zipfiles1/'
        fils=$(find $fp | grep cSize${cSize}_k${k}.npz)

        for f in ${fils[*]}; do
            python3 -c "import mrnaMI as mm; mm.convolveHalfPlane('$f',$mSize,axC=0);mm.convolveHalfPlane('$f',$mSize,axC=1);quit()"
        done
    elif [[ $1 == 3 ]]; then
        sd='/cis/home/kstouff4/Documents/SpatialTranscriptomics/Mouse/Mouse1_20220506/MI_Results/'
        mkdir $sd
        fp='/cis/home/kstouff4/Documents/SpatialTranscriptomics/Mouse/Mouse1_20220506/zipfiles1/'
        python3 -c "import mrnaMI as mm; mm.wholeBrainMI('$fp','$sd');quit()"

    fi
elif [[ $2 == 'merfish' ]]; then
    echo "computing merfish"
    fp='/cis/home/kstouff4/Documents/SpatialTranscriptomics/MERFISH/'
    fils=$(find $fp | grep gene | grep csv)
    if [[ $1 == 1 ]]; then
        for f in ${fils[*]}; do
            python3 -c "import mrnaMI as mm; mm.singleMI('$f',$cSize,$mSize,$k,'${f/gene/meta}'); quit()"
        done
    elif [[ $1 == 2 ]]; then
        fils=$(find $fp | grep cSize${cSize}_k${k}.npz)
        for f in ${fils[*]}; do
            python3 -c "import mrnaMI as mm; mm.convolveHalfPlane('$f',$mSize,axC=0);mm.convolveHalfPlane('$f',$mSize,axC=1);quit()"
        done
    elif [[ $1 == 3 ]]; then
        sd='/cis/home/kstouff4/Documents/SpatialTranscriptomics/MERFISH/MI_Results/'
        mkdir $sd
        python3 -c "import mrnaMI as mm; mm.wholeBrainMI('$fp','$sd');quit()"
    fi
fi
        
    
    
# copy all files from ax0 and ax1 into same folder to look at them collectively 


#dtCSV='/cis/home/kstouff4/Documents/SpatialTranscriptomics/Mouse/Mouse1_20220506/zipfiles1/202202231616_60988208_VMSC01001/detected_transcripts.csv'

#python3 -c "import mrnaMI as mm; mm.singleMI('$dtCSV',$cSize,$mSize,$k); quit()"