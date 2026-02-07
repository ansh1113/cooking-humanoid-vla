"""
Confusion matrix for robot primitives
"""
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from train_full_fidelity import FullFidelityVLA, CookingDataset

PRIMITIVE_MAP = {
    'stirring curry': 'STIR', 'stirring gently': 'STIR', 'sautéing': 'STIR',
    'frying onions': 'STIR', 'simmering': 'STIR', 'boiling': 'STIR',
    'roasting': 'STIR', 'mixing thoroughly': 'STIR', 'shallow frying': 'STIR',
    'deep frying': 'STIR',
    'adding water': 'POUR', 'adding oil': 'POUR', 'adding ghee': 'POUR',
    'adding butter': 'POUR', 'adding ginger garlic paste': 'POUR',
    'adding vegetables': 'TRANSFER', 'adding tomatoes': 'TRANSFER',
    'adding paneer': 'TRANSFER', 'adding chicken': 'TRANSFER',
    'adding rice': 'TRANSFER', 'plating': 'TRANSFER', 'serving': 'TRANSFER',
    'adding masala': 'SPRINKLE', 'adding turmeric': 'SPRINKLE',
    'adding chili powder': 'SPRINKLE', 'tempering spices': 'SPRINKLE',
    'garnishing with coriander': 'SPRINKLE', 'garnishing with cream': 'SPRINKLE',
    'adding coriander': 'SPRINKLE',
    'kneading dough': 'PRESS', 'grinding paste': 'PROCESS',
    'chopping onion': 'PROCESS', 'chopping vegetables': 'PROCESS',
    'mincing ginger garlic': 'PROCESS', 'peeling': 'PROCESS',
    'washing ingredients': 'PROCESS',
    'pressure cooking': 'WAIT', 'steaming': 'WAIT', 'grilling': 'WAIT'
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('data/processed/golden_40_dataset.json') as f:
    data = json.load(f)
with open('models/vocab_full.json') as f:
    vocab = json.load(f)

actions = [d['action'] for d in data]
_, val_data = train_test_split(data, test_size=0.15, stratify=actions, random_state=42)

val_loader = DataLoader(CookingDataset(val_data, vocab), batch_size=32)

checkpoint = torch.load('models/full_fidelity_vla.pt', map_location=device, weights_only=False)
model = FullFidelityVLA(len(vocab)).to(device)
if 'model' in checkpoint:
    model.load_state_dict(checkpoint['model'])
elif 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)
model.eval()

inv_vocab = {v: k for k, v in vocab.items()}

y_true_prim = []
y_pred_prim = []

with torch.no_grad():
    for feats, labels in val_loader:
        feats, labels = feats.to(device), labels.to(device)
        outputs = model(feats)
        _, preds = torch.max(outputs, 1)
        
        for true_lbl, pred_lbl in zip(labels.cpu().numpy(), preds.cpu().numpy()):
            true_act = inv_vocab[true_lbl]
            pred_act = inv_vocab[pred_lbl]
            y_true_prim.append(PRIMITIVE_MAP.get(true_act, 'UNKNOWN'))
            y_pred_prim.append(PRIMITIVE_MAP.get(pred_act, 'UNKNOWN'))

# Confusion matrix
primitives = sorted(list(set(y_true_prim)))
cm = confusion_matrix(y_true_prim, y_pred_prim, labels=primitives)

# Normalize
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
            xticklabels=primitives, yticklabels=primitives,
            cbar_kws={'label': 'Proportion'})
ax.set_xlabel('Predicted Primitive', fontsize=12)
ax.set_ylabel('True Primitive', fontsize=12)
ax.set_title('Robot Primitive Confusion Matrix (Normalized)\nValidation Set (65.7% Accuracy)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✅ Saved: confusion_matrix.png")
