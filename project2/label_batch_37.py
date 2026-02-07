"""
Label ONLY the 37 handpicked videos
"""
import sys
import os

# Change the input file in the labeling script
with open('label_golden_40_constrained_auto.py') as f:
    script = f.read()

# Replace the data file path
script = script.replace(
    "with open('golden_40_indian_videos.json')",
    "with open('batch_37_to_label.json')"
)

# Execute
exec(script)
