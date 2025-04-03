#!/bin/bash

# Get the absolute path of the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Export PYTHONPATH with this directory
export PYTHONPATH="$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR/sarathi-serve:$PYTHONPATH"

# Run the Python script with arguments
python "$SCRIPT_DIR/vidur/profiling/collectives/main.py" --collective all_reduce
