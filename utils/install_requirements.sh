#!/bin/bash 

#SBATCH --job-name=install-requirements
#SBATCH --time=10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=12G

source env/bin/activate
pip3 install torch torchvision torchaudio tensorboard 