#!/bin/bash

# Download the model from a cloud link if it is too large for GitHub
if [ ! -f model.pth ]; then
    echo "Downloading model..."
    wget -O model.pth 'your_cloud_link_here'
fi

# Run the seq2seq model
python3 model_seq2seq.py --data_dir "$1" --output_file "$2"





# #!/bin/bash

# # Check if the correct number of arguments is provided
# if [ "$#" -ne 2 ]; then
#     echo "Usage: ./hw2_seq2seq.sh <data_directory> <output_file>"
#     exit 1
# fi

# DATA_DIR=$1  # The first argument is the data directory
# OUTPUT_FILE=$2  # The second argument is the output file

# # Example: running the Python script to execute the seq2seq model
# python3 seq2seq_model.py --data_dir "$DATA_DIR" --output_file "$OUTPUT_FILE"

# # Notify user that the task has completed
# echo "Output saved to $OUTPUT_FILE"

