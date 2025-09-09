#! /bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: dqp_missing.sh <method> <dataest_path>"
    exit 1
fi

method=$1
dataset_path=$2

echo "Running with:\n    Method: $method\n    Dataset Path: $dataset_path"

pyenv local 3.11 && pyenv exec python3.11 /home/src/default_repo/utils/dqp_scripts/dqp_missing.py $method $dataset_path && pyenv local system && pyenv global system
