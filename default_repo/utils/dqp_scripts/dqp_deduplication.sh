#! /bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: dqp_deduplication.sh <config_path> <dataset_path>"
    exit 1
fi

config_path=$1
dataset_path=$2

pyenv local 3.11 && pyenv exec python3.11 /home/src/default_repo/utils/dqp_scripts/dqp_deduplication.py $config_path $dataset_path && pyenv local system && pyenv global system
