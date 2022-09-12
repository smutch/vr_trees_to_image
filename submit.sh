#!/bin/bash
#
#SBATCH --job-name=trees-to-image
#SBATCH --output=%x-%j-%a.out
#SBATCH --error=%x-%j-%a.err
#SBATCH --time=00:10:00
#SBATCH --array=0-31
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --mem=32G
#

module purge
module load $(tr '\n' ' ' << END
slurm/.latest
nvidia/.latest
gcccore/9.2.0
binutils/2.33.1
hwloc/2.0.3
git/2.18.0
gcc/9.2.0
openmpi/4.0.2
imkl/2019.5.281
szip/2.1.1
cuda/11.2.0
gsl/2.5
hdf5/1.10.5
fftw/3.3.8
END
)
printf "modules\n=======\n$(module -t list 2>&1)\n\n"

TREES_IN=/fred/oz113/N-bodys/L10_N2048_GridsHalos/trees/VELOCIraptor.walkabletree.forestID.hdf5.0
IMAGES_OUT=tree_images-$(printf "%03d" $SLURM_ARRAY_TASK_ID).h5

export RUST_LOG=info
./target/release/gen_tree_images -r $(printf "fds/fd.%03d" ${SLURM_ARRAY_TASK_ID}) $TREES_IN $IMAGES_OUT
