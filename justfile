set dotenv-load

input := env_var("TREES_IN")
output := env_var("IMAGES_OUT")

dev $RUST_LOG="debug":
    cargo run {{ input }} {{ output }}

run:
    cargo run --release {{ input }} {{ output }}

segment_fds splits:
    #!/usr/bin/env bash
    set -euxo pipefail

    tempdir=$(mktemp -d)
    trap 'rm -rf "$tempdir"' EXIT SIGINT SIGKILL

    cargo run --release -- -d {{ input }} ${tempdir}/fd_list.txt
    sort -n ${tempdir}/fd_list.txt > ${tempdir}/fd_sorted.txt
    split_size=$(awk 'END{ print int(NR/{{ splits }}) }' ${tempdir}/fd_sorted.txt)
    mkdir -p fds
    split -a3 -dl $split_size ${tempdir}/fd_sorted.txt fds/fd.


clean:
    cargo clean
    rm -r fds
