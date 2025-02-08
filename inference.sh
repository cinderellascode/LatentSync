#!/bin/bash

# Default super-resolution method
SUPERRES="none"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --superres)
            SUPERRES="$2"
            shift
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Run inference with super-resolution
python inference.py --superres $SUPERRES





