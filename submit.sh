#!/bin/bash
#
#SBATCH --job-name=trees-to-image
#SBATCH --output=%x-%j-%a.out
#SBATCH --error=%x-%j-%a.err
#SBATCH --time=00:15:00
#SBATCH --array=0-63
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --mem=16G
#

printf "\nmodules\n=======\n$(module -t list 2>&1)\n\n"

export TREES_IN=/fred/oz113/N-bodys/L10_N2048_GridsHalos/trees/VELOCIraptor.walkabletree.forestID.hdf5.0
export IMAGES_OUT=tree_images-${SLURM_ARRAY_TASK_ID}.h5

./target/release/gen_tree_images -r $(printf "fds/fd.%3d" ${SLURM_ARRAY_TASK_ID}) $TREES_IN $IMAGES_OUT
