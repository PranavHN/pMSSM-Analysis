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
echo 'export PATH="'${build_dir}'/softsusy-4.1.11/src/softsusy-build:$PATH"' >> $build_dir/setup.sh
echo 'export PATH="'${build_dir}'/micromegas_5.2.1/micromegas_5.2.1/MSSM:$PATH"' >> $build_dir/setup.sh
echo 'export PATH="'${build_dir}'/superiso_v4.0/src/superiso:$PATH"' >> $build_dir/setup.sh
echo 'export PATH="'${build_dir}'/GM2Calc-2.0.0/src/gm2calc-build/bin/:$PATH"' >> $build_dir/setup.sh
echo 'export PATH="'${build_dir}'/EVADE-0.2.0/src/evade-build/:$PATH"' >> $build_dir/setup.sh
echo 'export PYTHONPATH="'${build_dir}':$PYTHONPATH"' >> $build_dir/setup.sh
echo 'export DATAPATH="'${source_dir}/Run3ModelGen/data':$DATAPATH"' >> $build_dir/setup.sh
echo 'export EOSPATH="/eos/user/'$(whoami | head -c 1)/$(whoami)'/Run3ModelGen"' >> $build_dir/setup.sh
echo 'export RUN3MODELGENDIR="'${source_dir%source}'"' >> $build_dir/setup.sh