#!/bin/bash
cd $SLURM_SUBMIT_DIR
mkdir -p ../profile_data/$SLURM_JOB_NAME/
cp -f /dev/shm/profile.mon $SLURM_SUBMIT_DIR/../prof_data/$SLURM_JOB_NAME/profile_`hostname`.mon
rm -f /dev/shm/profile.mon

