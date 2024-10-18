#!/bin/bash

# Download the model from a cloud link if it is too large for GitHub
if [ ! -f model.pth ]; then
    echo "Downloading model..."
    wget -O model.pth 'your_cloud_link_here'
fi

# Run the seq2seq model
python3 model_seq2seq.py --data_dir "$1" --output_file "$2"
