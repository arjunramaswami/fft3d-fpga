#!/bin/bash

#SBATCH --partition=fpga --constraint=18.1.1_hpc
#SBATCH --time=30:00
#export VT_PCTRACE=1

module load nalla_pcie/18.1.1_hpc intelFPGA_pro/18.1.1_hpc

echo "Loaded 18.1.1_hpc"
cd $SLURM_SUBMIT_DIR/../

pwd
mkdir -p /dev/shm/profile/
cp $SLURM_SUBMIT_DIR/../bin/profile/fft3d.aocx /dev/shm/profile/
srun -o logs/%x.out -D /dev/shm $SLURM_SUBMIT_DIR/../bin/host -n $1 

#srun scripts/copy.sh
