#!/bin/bash

TARGET_DIR="data/VS13/MercedesAMG550"
FILE_NAME="MercedesAMG550.zip"
DOWNLOAD_URL="http://slobodan.ucg.ac.me/science/vs13/"

mkdir -p "data/VS13/"

if [ ! -d "$TARGET_DIR" ] || [ -z "$(ls -A "$TARGET_DIR" 2>/dev/null)" ]; then
    echo "Dataset folder is missing or empty: $TARGET_DIR"
    echo "Opening download page in default browser, download video + annotations for $FILE_NAME, extract them and place the folder in $TARGET_DIR"
    open -g "$DOWNLOAD_URL"

    echo "Press ENTER when download and extraction is complete"
    read -r

    if [ -d "$TARGET_DIR" ] && [ -n "$(ls -A "$TARGET_DIR" 2>/dev/null)" ]; then
        echo "Dataset ready in $TARGET_DIR"
    else
        echo "Dataset still not found in $TARGET_DIR"
        exit 1
    fi
else
    echo "Dataset ready in $TARGET_DIR"
fi