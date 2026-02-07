"""
STAGE 2: Execute Action Plan using REAL Project 1 Logic
Uses the actual PROJECT1_FINAL.py proven code
"""
import sys
import json
from pathlib import Path

# Import Project 1's COMPLETE proven system
sys.path.insert(0, '/root/vla_project/code')

# Need to define this for unpickling the model
class SimpleLanguageTokenizer:
    def __init__(self, vocab):
        self.word_to_idx = vocab
    
    def encode(self, instruction, max_len=10):
        words = instruction.lower().split()
        tokens = [self.word_to_idx.get(w, 1) for w in words[:max_len]]
        tokens += [0] * (max_len - len(tokens))
        return tokens

# Now import Project 1
import PROJECT1_FINAL as P1

def main():
    print("="*70)
    print("🎮 STAGE 2: EXECUTING WITH PROJECT 1 SYSTEM")
    print("="*70)
    
    # Load action plan
    print("\n📋 Loading action plan...")
    with open('action_plan.json', 'r') as f:
        plan_data = json.load(f)
    
    actions = plan_data['actions']
    print(f"   Video: {plan_data['video_title']}")
    print(f"   Actions: {len(actions)}")
    for i, a in enumerate(actions, 1):
        print(f"      {i}. {a['action']} ({a['confidence']:.3f})")
    
    # Load Project 1's VLA model
    print("\n📥 Loading Project 1 VLA model...")
    import torch
    import torchvision.transforms as transforms
    
    device = torch.device('cpu')
    model_path = "/root/vla_project/models_expanded/best_diverse_model.pt"
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    action_vocab = checkpoint['action_vocab']
    tokenizer = checkpoint['tokenizer']
    
    model = P1.FrozenResNetVLA(len(action_vocab), len(tokenizer.word_to_idx)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"   ✅ Model loaded with {len(action_vocab)} actions")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Launch controller with Project 1's setup
    print("\n🚀 Launching AI2-THOR with Project 1 setup...")
    controller = P1.launch_controller()
    
    # Map actions to commands
    action_to_command = {
        'PickupObject': 'pickup knife',
        'PutObject': 'place knife on counter',
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
    
    # Execute each action using Project 1's PROVEN logic
    print("\n" + "="*70)
    print("▶️  EXECUTING WITH PROJECT 1 SYSTEM")
    print("="*70)
    
    success_count = 0
    
    for i, action_data in enumerate(actions, 1):
        action = action_data['action']
        confidence = action_data['confidence']
        command = action_to_command.get(action, action.lower())
        
        print(f"\n[{i}/{len(actions)}] {action} ({confidence:.3f})")
        print(f"   Command: {command}")
        
        # Refresh state (Project 1's approach)
        P1.refresh_object_cache(controller)
        P1.WORLD_STATE.update_from_simulator(P1.ALL_OBJECTS)
        
        # Find target using Project 1's proven finder
        target_obj = P1.find_closest_object(controller, command)
        
        if not target_obj:
            print(f"   ❌ Object not found")
            continue
        
        print(f"   🎯 Target: {target_obj['objectType']}")
        
        # Check if action is redundant (Project 1's state tracking)
        should_skip, skip_reason = P1.WORLD_STATE.should_skip_action(action, target_obj)
        if should_skip:
            print(f"   ⏭️  SKIPPED: {skip_reason}")
            continue
        
        # Smart hand management (Project 1's logic)
        currently_holding = P1.get_held_object(controller)
        if currently_holding and P1.needs_empty_hand(action):
            print(f"   🧠 Dropping {currently_holding}")
            controller.step("DropHandObject")
        
        # Physics planning (Project 1's approach)
        ideal_dist, ideal_pitch, should_crouch = P1.calculate_physics_based_pose(target_obj)
        
        # Navigation (Project 1's proven navigation)
        nav_res = P1.get_best_nav_point(target_obj['position'], ideal_dist)
        if not nav_res:
            print(f"   ❌ Unreachable")
            continue
        
        best_pos, actual_dist = nav_res
        
        # Teleport (Project 1's approach)
        import math
        dx = target_obj['position']['x'] - best_pos['x']
        dz = target_obj['position']['z'] - best_pos['z']
        yaw = math.degrees(math.atan2(dx, dz))
        
        controller.step(
            "Teleport",
            position=best_pos,
            rotation=dict(x=0, y=yaw, z=0),
            horizon=ideal_pitch,
            standing=not should_crouch
        )
        
        # Execute with head scan (Project 1's proven execution)
        offsets = [0, 15, -15]
        success = False
        
        for off in offsets:
            if off != 0:
                if off > 0:
                    controller.step("LookDown", degrees=off)
                else:
                    controller.step("LookUp", degrees=abs(off))
            
            if action == "ThrowObject":
                event = controller.step(action="ThrowObject", moveMagnitude=10)
            elif action == "DropHandObject":
                event = controller.step(action="DropHandObject")
            else:
                event = controller.step(action=action, objectId=target_obj['objectId'])
            
            if event.metadata['lastActionSuccess']:
                print(f"   ✅ SUCCESS!")
                success = True
                success_count += 1
                
                # Update state
                P1.refresh_object_cache(controller)
                P1.WORLD_STATE.update_from_simulator(P1.ALL_OBJECTS)
                
                break
            
            if off != 0:
                if off > 0:
                    controller.step("LookUp", degrees=off)
                else:
                    controller.step("LookDown", degrees=abs(off))
        
        if not success:
            print(f"   ❌ FAILED: {event.metadata.get('errorMessage', 'Unknown')}")
    
    # Summary
    print("\n" + "="*70)
    print("📊 EXECUTION SUMMARY")
    print("="*70)
    print(f"Video: {plan_data['video_title']}")
    print(f"YouTube Action Plan: {' → '.join([a['action'] for a in actions])}")
    print(f"Total Actions: {len(actions)}")
    print(f"Successful: {success_count}/{len(actions)} ({success_count/len(actions)*100:.1f}%)")
    print("="*70)
    
    controller.stop()
    print("\n🎉 PROJECT 2 COMPLETE!")
    print("   YouTube Video → AI2-THOR Execution ✅")

if __name__ == "__main__":
    main()
