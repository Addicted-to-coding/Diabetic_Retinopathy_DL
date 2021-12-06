#!/bin/bash
#$ -M $USER@mail
#$ -m bea

. "/u/home/h/hbansal/anaconda3/etc/profile.d/conda.sh"

echo "Job $JOB_ID started on: " `hostname -s`
echo "Job $JOB_ID started on: " `date `
echo " "

MAIN_MODULE=$HOME/

JOB_DIR=$MAIN_MODULE/M226/simclr
cd $JOB_DIR

conda activate base 

python training.py --model=resnet --run=1 --epochs=50 --train=True --simclr=True --bsz=64 --pretrained=True
python training.py --model=resnet --run=2 --epochs=50 --train=True --simclr=True --bsz=64 --pretrained=True
python training.py --model=resnet --run=3 --epochs=50 --train=True --simclr=True --bsz=64 --pretrained=True
