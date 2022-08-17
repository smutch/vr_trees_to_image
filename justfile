python := "/Users/smutch/work/astro/projects/trees/trees_to_images/.env/bin/python"
input := "../../../../data/nbody/simulations/meraxes-validation/trees/VELOCIraptor.walkabletree.forestID.hdf5"
output := "tree_images.h5"

dev: && plot
    cargo run {{ input }} {{ output }}

run: && plot
    cargo run --release {{ input }} {{ output }}

plot:
    {{python}} ../plot.py
    open ./619000000001104.png
