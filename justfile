set dotenv-load

input := env_var("TREES_IN")
output := env_var("IMAGES_OUT")

dev $RUST_LOG="debug":
    cargo run --bin gen_tree_images {{ input }} {{ output }}

run:
    cargo run --release --bin gen_tree_images {{ input }} {{ output }}

segment_fds splits:
    #!/usr/bin/env bash
    set -euxo pipefail

    tempdir=$(mktemp -d)
    trap 'rm -rf "$tempdir"' EXIT SIGINT SIGKILL

    cargo run --release --bin gen_tree_images -- -d {{ input }} ${tempdir}/fd_list.txt
    sort -n ${tempdir}/fd_list.txt > ${tempdir}/fd_sorted.txt
    split_size=$(awk 'END{ print int(NR/{{ splits }}) }' ${tempdir}/fd_sorted.txt)
    mkdir -p fds
    split -a3 -dl $split_size ${tempdir}/fd_sorted.txt fds/fd.

run_segments splits:
    #!/usr/bin/env bash
    set -euxo pipefail
    cargo build --release --bin gen_tree_images
    for ((ii=0; ii<{{ splits }}; ii++)); do
        ii_str=$(printf "%03d" $ii)
        ./target/release/gen_tree_images -r fds/fd.${ii_str} {{ input }} $(echo {{ output }} | sed "s/\(.*\)\.\(.*\)/\1-${ii_str}.\2/")
    done

merge:
    cargo run --release --bin concat_output 'tree_images-*.h5' tree_images.h5

clean:
    cargo clean
    rm -r fds
