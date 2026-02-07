"""
Show dataset distribution
"""
import json
import matplotlib.pyplot as plt
from collections import Counter

with open('data/processed/golden_40_dataset.json') as f:
    data = json.load(f)

actions = [d['action'] for d in data]
counts = Counter(actions)

# Top 20 actions
top_20 = counts.most_common(20)
labels = [a for a, c in top_20]
values = [c for a, c in top_20]

fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.barh(range(len(labels)), values, color='steelblue')
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels)
ax.set_xlabel('Number of Samples', fontsize=12)
ax.set_title('Top 20 Cooking Actions Distribution (1,607 Total Samples)', 
             fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# Add counts on bars
for i, (bar, val) in enumerate(zip(bars, values)):
    ax.text(val + 2, i, str(val), va='center', fontsize=10)

plt.tight_layout()
plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
print("✅ Saved: class_distribution.png")
