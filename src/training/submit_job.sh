#!/bin/bash

#SBATCH --job-name=train_imagenet # Job name
#SBATCH --ntasks=1                # Number of tasks
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=0-16:00            # Runtime in D-HH:MM - takes about an hour per epoch on 2 GPUs
#SBATCH --partition=2080-galvani  # Partition to submit to
#SBATCH --mem=40G                 # Memory pool for all cores
#SBATCH --gres=gpu:1              # Request one GPU
#SBATCH --cpus-per-task=12        # Request 12 CPUs per GPU

# include information about the job in the output
scontrol show job=$SLURM_JOB_ID

# run the actual command
srun \
singularity exec \
--nv \
--bind /mnt/lustre/datasets/ \
--bind /mnt/lustre/work/bethge/tklein16/ \
--bind /scratch_local/ \
/mnt/lustre/work/bethge/tklein16/containers/hoyer.sif \
./train.sh

echo DONE.