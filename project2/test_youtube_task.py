"""
test_youtube_task.py - Test YouTube-learned task in Project 1 simulator
"""
import json

# Load task library
with open('data/processed/task_library.json', 'r') as f:
    library = json.load(f)

# Get Task_08 (PickupObject → SliceObject)
task_08 = [t for t in library['tasks'] if t['name'] == 'Task_08'][0]

print("="*70)
print("🧪 TESTING YOUTUBE-LEARNED TASK IN PROJECT 1")
print("="*70)
print(f"\nTask: {task_08['name']}")
print(f"Pattern: {task_08['pattern']}")
print(f"Source: Learned from {task_08['occurrences']} YouTube videos")
print("\nCommands to execute:")
for i, step in enumerate(task_08['steps'], 1):
    print(f"   {i}. {step['command']}")

print("\n" + "="*70)
print("📋 TO TEST IN PROJECT 1:")
print("="*70)
print("\n1. Run Project 1 interactive script:")
print("   cd ~/vla_project")
print("   apptainer exec --bind $PWD:/root/vla_project \\")
print("       containers/ai2thor_complete.sif \\")
print("       xvfb-run -a python3 /root/vla_project/code/PROJECT1_FINAL.py")
print("\n2. Execute these commands:")
for step in task_08['steps']:
    print(f"   {step['command']}")
print("\n3. Verify the learned sequence works!")
print("="*70)

# Also show Make_Salad
salad = [t for t in library['tasks'] if t['name'] == 'Make_Salad'][0]
print("\n🥗 BONUS: TEST COMPOSITE TASK")
print("="*70)
print(f"Task: {salad['name']}")
print(f"Description: {salad['description']}")
print("\nCommands:")
for i, step in enumerate(salad['steps'], 1):
    print(f"   {i}. {step['command']}")
print("="*70)
