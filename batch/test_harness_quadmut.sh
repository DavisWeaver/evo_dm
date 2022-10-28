#!/bin/bash
#SBATCH --time=200:00:00
#SBATCH -N 1 --mem=100gb
#SBATCH --output=test_harness_mut.out
#SBATCH --mail-user=dtw43@case.edu
#SBATCH --mail-type=FAIL
#SBATCH --job-name=test_harness_quadmut

# CD to the right directory
cd ~/RL_vs_Resistance

#create a temp directory
tempdir="$(mktemp -d)"
env_name="/evoldm_env"
module load gcc
module load openmpi
module load python/3.7.0
python -m venv $tempdir$env_name
source "$tempdir$env_name/bin/activate"
pip install --upgrade pip
pip install git+https://github.com/DavisWeaver/evo_dm.git

module load R
Rscript R/test_harness_quadmut.R
