#!/bin/bash
#SBATCH -A pc2-mitarbeiter
#SBATCH --partition=long
#SBATCH -N 1
#SBATCH --time=1-00:00:00

cd $SLURM_SUBMIT_DIR
cd ..

module load intelFPGA_pro/18.1.1_hpc nalla_pcie/18.1.1_hpc

echo "Finding makefile in $pwd"

if [ "$1" == "syn" ]; then    
    echo "Synthesizing " 
    make syn
elif [ "$1" == "profile" ]; then
    echo "Profiling"
    make profile
elif [ -z "$1" ]; then
    echo "No Args found. Pass relevant arguments"
else
    echo "Wrong arguments"
fi

#aoc -fp-relaxed -v -g device/fft3d.cl -board=p520_hpc_sg280l -o bin/synthesis/fft3d.aocx > test.log  


