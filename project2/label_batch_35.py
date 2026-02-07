import sys
with open('label_golden_40_constrained_auto.py') as f:
    script = f.read()
script = script.replace(
    "with open('golden_40_indian_videos.json')",
    "with open('batch_35_to_label.json')"
)
exec(script)
