"""
📊 REALISTIC EVALUATION (FIXED)
Measures "Robot-Ready" Accuracy (Top-3 & Primitive Match)
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

# VERB MAPPING (Collapsing ingredients for scoring only)
def get_verb(action):
    # Maps "adding water" -> "adding"
    if "adding" in action: return "adding"
    if "stirring" in action: return "stirring"
    if "chopping" in action: return "chopping"
    if "garnishing" in action: return "garnishing"
    if "mixing" in action: return "mixing"
    if "kneading" in action: return "kneading"
    return action.split(' ')[0] # Default to first word

def main():
    print("🚀 RUNNING DIAGNOSTIC EVALUATION...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Data
    with open(DATA_FILE) as f: data = json.load(f)
    with open(VOCAB_PATH) as f: vocab = json.load(f)
    
    # Re-create Split (Must match training random_state=42)
    actions = [d['action'] for d in data]
    _, val_data = train_test_split(data, test_size=0.15, stratify=actions, random_state=42)
    
    val_ds = CookingDataset(val_data, vocab, augment=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    
    # Load Model
    # FIXED: Using 'model' key instead of 'model_state_dict'
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = FullFidelityVLA(len(vocab)).to(device)
    
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Fallback: assume checkpoint IS the state dict
        model.load_state_dict(checkpoint)
        
    model.eval()
    
    inv_vocab = {v: k for k, v in vocab.items()}
    
    # Metrics
    correct_top1 = 0
    correct_top3 = 0
    correct_verb = 0
    total = 0
    
    print(f"📝 Evaluating {len(val_data)} validation samples...")
    
    with torch.no_grad():
        for feats, labels in val_loader:
            feats, labels = feats.to(device), labels.to(device)
            outputs = model(feats)
            
            # Top-k
            _, top3_preds = outputs.topk(3, 1, largest=True, sorted=True)
            top1_preds = top3_preds[:, 0]
            
            for i in range(labels.size(0)):
                true_label = labels[i].item()
                pred_label = top1_preds[i].item()
                top3_labels = top3_preds[i].tolist()
                
                # 1. Exact Match
                if pred_label == true_label:
                    correct_top1 += 1
                    
                # 2. Top-3 Match
                if true_label in top3_labels:
                    correct_top3 += 1
                    
                # 3. Verb Match (Primitive)
                true_act = inv_vocab[true_label]
                pred_act = inv_vocab[pred_label]
                
                # Get the core verb (e.g., "adding" from "adding oil")
                if get_verb(true_act) == get_verb(pred_act):
                    correct_verb += 1
                    
                total += 1

    print("\n" + "="*50)
    print("🏆 FINAL ROBOTIC SCORES")
    print("="*50)
    print(f"❌ Strict Accuracy (Exact Class):  {100 * correct_top1 / total:.1f}%")
    print(f"⚠️  Top-3 Accuracy (Forgiving):    {100 * correct_top3 / total:.1f}%")
    print(f"✅ Primitive Accuracy (Verb Only): {100 * correct_verb / total:.1f}%")
    print("-" * 50)
    
    acc_verb = 100 * correct_verb / total
    if acc_verb > 70:
        print(f"🎉 VERDICT: PASSED ({acc_verb:.1f}%). The robot knows WHAT to do.")
    else:
        print("⚠️  VERDICT: NEEDS IMPROVEMENT.")

if __name__ == "__main__":
    main()
