#!/bin/zsh

# This assumes we used venv to create a virtual environment
source venv/bin/activate

# Use time to measure how long the process takes
time python process.py ./decimator_2023_12_18 "2023-12-18 01:00" "2023-12-18 19:00" ./out/ --jobs 3
