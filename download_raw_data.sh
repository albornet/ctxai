#!/bin/bash

set -e

# Function to check if a command is available
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Check if 'parallel' is installed
if ! command_exists parallel; then
  echo "'parallel' is not installed. Please install it using:"
  echo "  sudo apt-get install parallel   # On Ubuntu/Debian"
  echo "  brew install parallel           # On macOS"
  exit 1
fi

# Check if 'pv' is installed
if ! command_exists pv; then
  echo "'pv' is not installed. Please install it using:"
  echo "  sudo apt-get install pv         # On Ubuntu/Debian"
  echo "  brew install pv                 # On macOS"
  exit 1
fi

# Define paths
DEST_DIR="./data_ctgov/raw_files"
ZIP_FILE="./data_ctgov/raw_files/data.zip"

# Create directories
mkdir -p "$DEST_DIR"

# Download a zipped version of clinical trial json files
echo "Downloading ClinicalTrial.gov data to $ZIP_FILE..."
wget -O "$ZIP_FILE" "https://clinicaltrials.gov/api/int/studies/download?format=json.zip&start=2000-01-01_2024-06-01&aggFilters=phase%3A1+2+3+4%2CstudyType%3Aint"

# Verify the file was downloaded
if [[ ! -f "$ZIP_FILE" ]]; then
  echo "Download failed!"
  exit 1
fi
echo "Download successful!"

# Function to handle each file during extraction
extract_and_move() {
  local file_path=$1
  local file_name=$(basename "$file_path")
  local subdir="${file_name:0:7}xxxx"
  local dest_dir="$DEST_DIR/$subdir"

  mkdir -p "$dest_dir"
  mv "$file_path" "$dest_dir"
}

export DEST_DIR
export -f extract_and_move

# Create a temporary directory for extraction
TMP_DIR=$(mktemp -d)
echo "Unzipping $ZIP_FILE to a temporary directory..."
unzip -q "$ZIP_FILE" -d "$TMP_DIR"
echo "Unzipping successful!"

# Count the total number of files to process
TOTAL_FILES=$(find "$TMP_DIR" -type f | wc -l)

# Process each file in the temporary directory using GNU Parallel and pv for the progress bar
echo "Moving unzipped clinical trials to dedicated subdirectories"
find "$TMP_DIR" -type f | pv -l -s $TOTAL_FILES | parallel extract_and_move {}

# Cleanup
rm -r "$TMP_DIR"
rm "$ZIP_FILE"

echo "Processing completed successfully!"
