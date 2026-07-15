#!/bin/bash

# Initialize variables to hold the values
source_dir=""
build_dir=""

# Parse command-line options
while getopts 's:b:' flag; do
    case "${flag}" in
        s) source_dir="${OPTARG}" ;;
        b) build_dir="${OPTARG}" ;;
        *) echo "Usage: $0 -s <source_dir> -b <build_dir>" >&2
           exit 1 ;;
    esac
done

# Ensure both values are provided
if [[ -z "$source_dir" || -z "$build_dir" ]]; then
    echo "Both source and build dir need to be provided."
    exit 1
fi

# Define a function to ignore errors
handle_error() {
    echo "WARNING: some files in $build_dir already existed. Is this intended?"
    # Additional error handling code here
    # exit 1
}

# Move setup script to build dir
cp $source_dir/setup.sh $build_dir

# Activate custom error handling
trap 'handle_error' ERR

# Create build dir and EOS dirs
mkdir $build_dir/Run3ModelGen 2>/dev/null
mkdir -p /eos/user/$(whoami | head -c 1)/$(whoami)/Run3ModelGen 2>/dev/null

# Link python modules in build
ln -s $source_dir/Run3ModelGen/python/* $build_dir/Run3ModelGen/ 2>/dev/null

# Deactivate custom error handling
trap - ERR

# Manipulating env vars: PATH, PYTHONPATH and DATAPATH
echo 'export PATH="'${source_dir}'/Run3ModelGen/scripts:$PATH"' >> $build_dir/setup.sh
echo 'export PATH="'${build_dir}'/micromegas_5.2.1/src/micromegas/MSSM:$PATH"' >> $build_dir/setup.sh
# echo 'export PATH="'${build_dir}'/EVADE-0.2.0/src/evade-build/:$PATH"' >> $build_dir/setup.sh
echo 'export PYTHONPATH="'${build_dir}':$PYTHONPATH"' >> $build_dir/setup.sh
echo 'export DATAPATH="'${source_dir}/Run3ModelGen/data':$DATAPATH"' >> $build_dir/setup.sh
echo 'export EOSPATH="/eos/user/'$(whoami | head -c 1)/$(whoami)'/Run3ModelGen"' >> $build_dir/setup.sh
echo 'export RUN3MODELGENDIR="'${source_dir%source}'"' >> $build_dir/setup.sh
echo '# Check if eos directory is available & set SMODELS_CACHEDIR accordingly' >> $build_dir/setup.sh
echo 'if [ -d /eos/user/'$(whoami | head -c 1)/$(whoami)'/Run3ModelGen ]; then' >> $build_dir/setup.sh
echo '  export SMODELS_CACHEDIR="/eos/user/'$(whoami | head -c 1)/$(whoami)'/Run3ModelGen"' >> $build_dir/setup.sh
echo 'else' >> $build_dir/setup.sh >> $build_dir/setup.sh
echo '  export SMODELS_CACHEDIR="'${source_dir%source}'"' >> $build_dir/setup.sh
echo 'fi' >> $build_dir/setup.sh >> $build_dir/setup.sh

# Ensure SModelS's bundled pythia8 is built. This runs on every `source build/setup.sh`,
# so it also covers jobs that start from a fresh pixi env where the build-time pre-build
# didn't persist (e.g. CI genModels jobs re-create the env via `pixi shell-hook`, and
# only `build/` — not `.pixi/` — is passed between CI stages). Idempotent: a ~0.4s no-op
# once pythia8 is built. Run as a subshell (not sourced) so its exit can't kill the
# sourcing shell, and non-fatal so sourcing setup.sh never breaks.
echo '# Ensure SModelS pythia8 is built (idempotent; workaround for broken auto-download)' >> $build_dir/setup.sh
echo 'bash "'${source_dir}'/Run3ModelGen/scripts/setup_pythia8.sh" || echo "[setup.sh] WARNING: pythia8 setup failed; the SModelS step may not run"' >> $build_dir/setup.sh
