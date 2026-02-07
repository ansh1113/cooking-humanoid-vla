"""
📊 FINAL EVALUATION (CORRECTED)
Measures accuracy based on ROBOT PRIMITIVES, not string matching.
"""
import torch
from torch.utils.data import DataLoader
from train_full_fidelity import FullFidelityVLA, CookingDataset
import json
import os
import numpy as np
from sklearn.model_selection import train_test_split

# CONFIG
FEATURE_DIR = 'data/features_golden40'
DATA_FILE = 'data/processed/golden_40_dataset.json'
MODEL_PATH = 'models/full_fidelity_vla.pt'
VOCAB_PATH = 'models/vocab_full.json'
BATCH_SIZE = 32

# THE CRITIC'S CORRECTED MAPPING
# Maps 40 Fine-Grained Classes -> 8 Robot Primitives
PRIMITIVE_MAP = {
    # STIR group
    'stirring curry': 'STIR', 'stirring gently': 'STIR', 'sautéing': 'STIR',
    'frying onions': 'STIR', 'simmering': 'STIR', 'boiling': 'STIR',
    'roasting': 'STIR', 'mixing thoroughly': 'STIR', 'shallow frying': 'STIR',
    'deep frying': 'STIR',
    
    # POUR group
    'adding water': 'POUR', 'adding oil': 'POUR', 'adding ghee': 'POUR',
    'adding butter': 'POUR', 'adding ginger garlic paste': 'POUR',
    
    # TRANSFER group
    'adding vegetables': 'TRANSFER', 'adding tomatoes': 'TRANSFER',
    'adding paneer': 'TRANSFER', 'adding chicken': 'TRANSFER',
    'adding rice': 'TRANSFER', 'plating': 'TRANSFER', 'serving': 'TRANSFER',
    
    # SPRINKLE group
    'adding masala': 'SPRINKLE', 'adding turmeric': 'SPRINKLE',
    'adding chili powder': 'SPRINKLE', 'tempering spices': 'SPRINKLE',
    'garnishing with coriander': 'SPRINKLE', 'garnishing with cream': 'SPRINKLE',
    'adding coriander': 'SPRINKLE',
    
    # MANIPULATION group
    'kneading dough': 'PRESS', 'grinding paste': 'PROCESS',
    'chopping onion': 'PROCESS', 'chopping vegetables': 'PROCESS',
    'mincing ginger garlic': 'PROCESS', 'peeling': 'PROCESS',
    'washing ingredients': 'PROCESS',
    
    # PASSIVE group
    'pressure cooking': 'WAIT', 'steaming': 'WAIT', 'grilling': 'WAIT'
}

def get_primitive(action):
    return PRIMITIVE_MAP.get(action, 'UNKNOWN')

def main():
    print("🚀 RUNNING FINAL METRIC EVALUATION...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open(DATA_FILE) as f: data = json.load(f)
    with open(VOCAB_PATH) as f: vocab = json.load(f)
    
    # Re-create Split
    actions = [d['action'] for d in data]
    _, val_data = train_test_split(data, test_size=0.15, stratify=actions, random_state=42)
    
    val_loader = DataLoader(CookingDataset(val_data, vocab, augment=False), batch_size=BATCH_SIZE)
    
    # Load Model
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = FullFidelityVLA(len(vocab)).to(device)
    if 'model' in checkpoint: model.load_state_dict(checkpoint['model'])
    elif 'model_state_dict' in checkpoint: model.load_state_dict(checkpoint['model_state_dict'])
    else: model.load_state_dict(checkpoint)
    model.eval()
    
    inv_vocab = {v: k for k, v in vocab.items()}
    
    correct_top1 = 0
    correct_top3 = 0
    correct_prim = 0
    total = 0
    
    print(f"📝 Evaluating {len(val_data)} validation samples...")
    
    with torch.no_grad():
        for feats, labels in val_loader:
            feats, labels = feats.to(device), labels.to(device)
            outputs = model(feats)
            _, top3_preds = outputs.topk(3, 1, largest=True, sorted=True)
            top1_preds = top3_preds[:, 0]
            
            for i in range(labels.size(0)):
                true_lbl = labels[i].item()
                pred_lbl = top1_preds[i].item()
                
                # 1. Strict
                if pred_lbl == true_lbl: correct_top1 += 1
                
                # 2. Top-3
                if true_lbl in top3_preds[i].tolist(): correct_top3 += 1
                
                # 3. Primitive (The Critic's Fix)
                true_act = inv_vocab[true_lbl]
                pred_act = inv_vocab[pred_lbl]
                if get_primitive(true_act) == get_primitive(pred_act):
                    correct_prim += 1
                    
                total += 1

    print("\n" + "="*50)
    print("🏆 FINAL CORRECTED SCORES")
    print("="*50)
    print(f"❌ Strict Accuracy:     {100 * correct_top1 / total:.1f}%")
    print(f"⚠️ Top-3 Accuracy:      {100 * correct_top3 / total:.1f}%")
    print(f"✅ Primitive Accuracy:  {100 * correct_prim / total:.1f}%")
    print("-" * 50)
    
    if (100 * correct_prim / total) > 75:
        print("🎉 VERDICT: A+ (Deployable)")
    elif (100 * correct_prim / total) > 70:
        print("✅ VERDICT: A- (Strong)")
    else:
        print("⚠️ VERDICT: Needs Tuning")

if __name__ == "__main__":
    main()
