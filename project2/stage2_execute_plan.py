"""
STAGE 2: Execute Action Plan in AI2-THOR
Runs inside Apptainer container with full Project 1 logic
"""
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import ai2thor.controller
from PIL import Image
import json
import math
import time

# Project 1 VLA Model
class FrozenResNetVLA(nn.Module):
    def __init__(self, num_actions, vocab_size, hidden_dim=512):
        super().__init__()
        resnet = models.resnet50(weights=None)
        self.vision_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.vision_proj = nn.Sequential(
            nn.Linear(2048, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(0.3)
        )
        self.language_embedding = nn.Embedding(vocab_size, 128)
        self.language_lstm = nn.LSTM(128, hidden_dim // 2, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=0.3, batch_first=True)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_actions)
        )
    
    def forward(self, images, language):
        vision_features = self.vision_encoder(images).squeeze(-1).squeeze(-1)
        vision_embed = self.vision_proj(vision_features)
        lang_embed = self.language_embedding(language)
        lang_output, _ = self.language_lstm(lang_embed)
        lang_features = lang_output[:, -1, :]
        lang_features = torch.cat([lang_features, lang_features], dim=-1)
        vision_query = vision_embed.unsqueeze(1)
        lang_key = lang_features.unsqueeze(1)
        attn_out, _ = self.attention(vision_query, lang_key, lang_key)
        fused = torch.cat([attn_out.squeeze(1), lang_features], dim=-1)
        return self.action_head(fused)

class SimpleLanguageTokenizer:
    def __init__(self, vocab):
        self.word_to_idx = vocab
    
    def encode(self, instruction, max_len=10):
        words = instruction.lower().split()
        tokens = [self.word_to_idx.get(w, 1) for w in words[:max_len]]
        tokens += [0] * (max_len - len(tokens))
        return tokens

# Action to command mapping
ACTION_COMMANDS = {
    'PickupObject': 'pickup knife',
    'PutObject': 'place knife',
    'SliceObject': 'slice the tomato',
    'CookObject': 'cook the egg',
    'OpenObject': 'open fridge',
    'CloseObject': 'close fridge',
    'ToggleObjectOn': 'turn on stove',
    'ToggleObjectOff': 'turn off stove',
    'DropHandObject': 'drop item',
    'ThrowObject': 'throw item',
    'BreakObject': 'crack egg'
}

REACHABLE_POINTS = []
ALL_OBJECTS = []

def launch_controller():
    global REACHABLE_POINTS, ALL_OBJECTS
    
    print("   🔌 Starting AI2-THOR...")
    c = ai2thor.controller.Controller(
        scene="FloorPlan1", 
        gridSize=0.05, 
        width=600, 
        height=600, 
        fieldOfView=100, 
        platform="Linux64"
    )
    
    print("   🗺️  Caching NavMesh...")
    REACHABLE_POINTS = c.step("GetReachablePositions").metadata['actionReturn']
    ALL_OBJECTS = c.last_event.metadata['objects']
    print("   ✅ Ready!")
    
    return c

def find_object(controller, obj_type):
    """Simple object finder"""
    for obj in ALL_OBJECTS:
        if obj_type.lower() in obj['objectType'].lower() and obj['visible']:
            return obj
    return None

def navigate_to_object(controller, target_obj):
    """Simple navigation"""
    agent_pos = controller.last_event.metadata['agent']['position']
    target_pos = target_obj['position']
    
    # Find nearby reachable point
    best_pos = None
    best_dist = float('inf')
    
    for pos in REACHABLE_POINTS:
        dist = math.sqrt((target_pos['x']-pos['x'])**2 + (target_pos['z']-pos['z'])**2)
        if 0.5 < dist < 1.2 and dist < best_dist:
            best_pos = pos
            best_dist = dist
    
    if not best_pos:
        return False
    
    # Navigate
    dx = target_pos['x'] - best_pos['x']
    dz = target_pos['z'] - best_pos['z']
    yaw = math.degrees(math.atan2(dx, dz))
    
    controller.step(
        "Teleport",
        position=best_pos,
        rotation=dict(x=0, y=yaw, z=0),
        horizon=0,
        standing=True
    )
    
    return True

def execute_action(controller, action, target_obj):
    """Execute single action"""
    if action in ['ThrowObject', 'DropHandObject']:
        event = controller.step(action=action)
    else:
        event = controller.step(action=action, objectId=target_obj['objectId'])
    
    return event.metadata['lastActionSuccess']

def main():
    print("="*70)
    print("🎮 STAGE 2: EXECUTING IN AI2-THOR")
    print("="*70)
    
    # Load action plan
    print("\n📋 Loading action plan...")
    with open('action_plan.json', 'r') as f:
        plan_data = json.load(f)
    
    actions = plan_data['actions']
    print(f"   Video: {plan_data['video_title']}")
    print(f"   Actions: {len(actions)}")
    
    # Launch simulator
    print("\n🚀 Launching simulator...")
    controller = launch_controller()
    
    # Execute each action
    print("\n" + "="*70)
    print("▶️  EXECUTING ACTIONS")
    print("="*70)
    
    success_count = 0
    
    for i, action_data in enumerate(actions, 1):
        action = action_data['action']
        confidence = action_data['confidence']
        command = ACTION_COMMANDS.get(action, action)
        
        print(f"\n[{i}/{len(actions)}] {action} ({confidence:.3f})")
        print(f"   Command: {command}")
        
        # Find target object
        if action == 'SliceObject':
            target = find_object(controller, 'tomato')
        elif action == 'PickupObject':
            target = find_object(controller, 'knife')
        elif action == 'CookObject':
            target = find_object(controller, 'egg')
        elif action == 'BreakObject':
            target = find_object(controller, 'egg')
        elif action == 'OpenObject':
            target = find_object(controller, 'fridge')
        elif action == 'CloseObject':
            target = find_object(controller, 'fridge')
        elif action in ['PutObject', 'DropHandObject', 'ThrowObject']:
            target = None  # No target needed
        else:
            target = None
        
        if action not in ['PutObject', 'DropHandObject', 'ThrowObject'] and not target:
            print(f"   ❌ Target not found")
            continue
        
        # Navigate
        if target:
            print(f"   🎯 Target: {target['objectType']}")
            nav_success = navigate_to_object(controller, target)
            if not nav_success:
                print(f"   ❌ Navigation failed")
                continue
        
        # Execute
        success = execute_action(controller, action, target)
        
        if success:
            print(f"   ✅ SUCCESS!")
            success_count += 1
        else:
            print(f"   ❌ FAILED")
        
        time.sleep(0.5)
    
    # Summary
    print("\n" + "="*70)
    print("📊 EXECUTION SUMMARY")
    print("="*70)
    print(f"Total Actions: {len(actions)}")
    print(f"Successful: {success_count}/{len(actions)} ({success_count/len(actions)*100:.1f}%)")
    print("="*70)
    
    controller.stop()
    print("\n🎉 EXECUTION COMPLETE!")

if __name__ == "__main__":
    main()
