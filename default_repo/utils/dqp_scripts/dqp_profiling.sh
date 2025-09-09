#! /bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: dqp_deduplication.sh <dataset_path>"
    exit 1
fi

dataset_path=$1

pyenv local 3.11 && pyenv exec python3.11 /home/src/default_repo/utils/dqp_scripts/dqp_profiling.py $dataset_path && pyenv local system && pyenv global system
