"""
System architecture flowchart
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(5, 9.5, 'Indian Cooking VLA System Architecture', 
        ha='center', fontsize=18, fontweight='bold')

# Input
box1 = FancyBboxPatch((0.5, 7.5), 2, 1, boxstyle="round,pad=0.1", 
                       facecolor='lightblue', edgecolor='black', linewidth=2)
ax.add_patch(box1)
ax.text(1.5, 8, 'YouTube Video\nURL', ha='center', va='center', fontsize=11, fontweight='bold')

# Whisper
arrow1 = FancyArrowPatch((2.5, 8), (3.5, 8), arrowstyle='->', 
                         mutation_scale=20, linewidth=2, color='black')
ax.add_patch(arrow1)
box2 = FancyBboxPatch((3.5, 7.5), 2, 1, boxstyle="round,pad=0.1", 
                       facecolor='#FFE5B4', edgecolor='black', linewidth=2)
ax.add_patch(box2)
ax.text(4.5, 8, 'Whisper\n(Audio)', ha='center', va='center', fontsize=11, fontweight='bold')

# CLIP
box3 = FancyBboxPatch((3.5, 6), 2, 1, boxstyle="round,pad=0.1", 
                       facecolor='#FFE5B4', edgecolor='black', linewidth=2)
ax.add_patch(box3)
ax.text(4.5, 6.5, 'CLIP ViT-B/32\n(Vision)', ha='center', va='center', fontsize=11, fontweight='bold')

arrow2 = FancyArrowPatch((2.5, 7.8), (3.5, 6.7), arrowstyle='->', 
                         mutation_scale=20, linewidth=2, color='black')
ax.add_patch(arrow2)

# VLA Model
arrow3 = FancyArrowPatch((5.5, 8), (6.5, 6.5), arrowstyle='->', 
                         mutation_scale=20, linewidth=2, color='black')
ax.add_patch(arrow3)
arrow4 = FancyArrowPatch((5.5, 6.5), (6.5, 6.5), arrowstyle='->', 
                         mutation_scale=20, linewidth=2, color='black')
ax.add_patch(arrow4)

box4 = FancyBboxPatch((6.5, 5.5), 2.5, 2, boxstyle="round,pad=0.1", 
                       facecolor='#90EE90', edgecolor='black', linewidth=3)
ax.add_patch(box4)
ax.text(7.75, 6.8, 'VLA Transformer', ha='center', fontsize=12, fontweight='bold')
ax.text(7.75, 6.4, '6 Layers, 512 Hidden', ha='center', fontsize=9)
ax.text(7.75, 6.1, '19M Parameters', ha='center', fontsize=9)

# Hierarchical Mapping
arrow5 = FancyArrowPatch((7.75, 5.5), (7.75, 4.5), arrowstyle='->', 
                         mutation_scale=20, linewidth=2, color='black')
ax.add_patch(arrow5)
box5 = FancyBboxPatch((6, 3.5), 3.5, 1, boxstyle="round,pad=0.1", 
                       facecolor='#FFB6C1', edgecolor='black', linewidth=2)
ax.add_patch(box5)
ax.text(7.75, 4, 'Hierarchical Mapping\n40 Actions → 8 Primitives', 
        ha='center', va='center', fontsize=10, fontweight='bold')

# Output
arrow6 = FancyArrowPatch((7.75, 3.5), (7.75, 2.5), arrowstyle='->', 
                         mutation_scale=20, linewidth=2, color='black')
ax.add_patch(arrow6)
box6 = FancyBboxPatch((6, 1.5), 3.5, 1, boxstyle="round,pad=0.1", 
                       facecolor='lightcoral', edgecolor='black', linewidth=2)
ax.add_patch(box6)
ax.text(7.75, 2, 'Robot Commands\nStir(), Pour(), Transfer()...', 
        ha='center', va='center', fontsize=11, fontweight='bold')

# Stats box
stats_box = FancyBboxPatch((0.5, 0.2), 3, 2.5, boxstyle="round,pad=0.1", 
                           facecolor='lightyellow', edgecolor='black', linewidth=2)
ax.add_patch(stats_box)
ax.text(2, 2.3, 'Key Statistics', ha='center', fontsize=11, fontweight='bold', 
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax.text(2, 1.9, '• 1,607 training samples', ha='center', fontsize=9)
ax.text(2, 1.6, '• 40 fine-grained actions', ha='center', fontsize=9)
ax.text(2, 1.3, '• 77% demo accuracy', ha='center', fontsize=9)
ax.text(2, 1.0, '• 90%+ on core primitives', ha='center', fontsize=9)
ax.text(2, 0.7, '• 65.7% primitive accuracy', ha='center', fontsize=9)
ax.text(2, 0.4, '• 19M parameters', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('system_architecture.png', dpi=300, bbox_inches='tight')
print("✅ Saved: system_architecture.png")
