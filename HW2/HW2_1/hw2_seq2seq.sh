#!/bin/bash


if [ "$#" -ne 2 ]; then
    echo "Usage: ./hw2_seq2seq.sh <data_directory> <output_file>"
    exit 1
fi

DATA_DIR=$1
OUTPUT_FILE=$2

python3 seq2seq_model.py "$DATA_DIR" "$OUTPUT_FILE"
