#!/usr/bin/env bash

files=$(find /mnt/storage/turn-taking-projects/corpora/switchboard/switchboard_wav -name "*.wav")

for file in $files; do
    
    python gemaps.py --target_sr 16000 --stereo 1 --input_file $file

done