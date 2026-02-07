"""
create_task_library.py - Convert discovered sequences into executable tasks
Creates a library of composite tasks for Project 1
"""
import json
from pathlib import Path

# Map actions to Project 1 natural language commands
ACTION_TEMPLATES = {
    'PickupObject': [
        'pickup {object}',
        'grab the {object}',
        'take the {object}'
    ],
    'PutObject': [
        'place {object}',
        'put {object} on countertop',
        'set down {object}'
    ],
    'SliceObject': [
        'slice the {object}',
        'cut the {object}',
        'chop the {object}'
    ],
    'CookObject': [
        'cook the {object}',
        'heat the {object}'
    ],
    'OpenObject': [
        'open the {object}'
    ],
    'CloseObject': [
        'close the {object}'
    ],
    'ToggleObjectOn': [
        'turn on the {object}'
    ],
    'ToggleObjectOff': [
        'turn off the {object}'
    ]
}

# Common cooking objects for each action
OBJECT_MAPPINGS = {
    'PickupObject': ['knife', 'apple', 'tomato', 'lettuce', 'mug', 'pan'],
    'PutObject': ['knife', 'apple', 'mug'],
    'SliceObject': ['tomato', 'lettuce', 'potato', 'apple', 'bread'],
    'CookObject': ['egg', 'potato'],
    'OpenObject': ['fridge', 'microwave'],
    'CloseObject': ['fridge', 'microwave'],
    'ToggleObjectOn': ['stove', 'microwave'],
    'ToggleObjectOff': ['stove', 'microwave']
}

def pattern_to_task(pattern, task_name):
    """Convert action pattern to executable task"""
    
    actions = pattern.split(' → ')
    
    # Generate command sequence
    commands = []
    for i, action in enumerate(actions):
        if action in ACTION_TEMPLATES and action in OBJECT_MAPPINGS:
            # Pick appropriate object
            objects = OBJECT_MAPPINGS[action]
            obj = objects[i % len(objects)]  # Cycle through objects
            
            # Pick first template
            template = ACTION_TEMPLATES[action][0]
            command = template.format(object=obj)
            commands.append({
                'action': action,
                'command': command,
                'object': obj
            })
    
    return {
        'name': task_name,
        'pattern': pattern,
        'steps': commands,
        'description': f"Execute sequence: {pattern}"
    }

def create_task_library():
    """Create library of executable tasks"""
    
    print("="*70)
    print("📚 CREATING TASK LIBRARY")
    print("="*70)
    
    # Load sequences
    with open('data/processed/action_sequences.json', 'r') as f:
        sequences_data = json.load(f)
    
    common_patterns = sequences_data['common_patterns']
    
    print(f"\n📋 Converting {len(common_patterns)} patterns to tasks...")
    
    task_library = []
    
    for i, (pattern, data) in enumerate(sorted(common_patterns.items(), key=lambda x: -x[1]['count'])):
        task_name = f"Task_{i+1:02d}"
        task = pattern_to_task(pattern, task_name)
        task['occurrences'] = data['count']
        
        task_library.append(task)
        
        print(f"\n{task_name}: {pattern} ({data['count']}x)")
        print(f"   Commands:")
        for step in task['steps']:
            print(f"      {step['command']}")
    
    # Add some hand-crafted composite tasks
    print("\n📝 Adding hand-crafted composite tasks...")
    
    composite_tasks = [
        {
            'name': 'Make_Salad',
            'pattern': 'PickupObject → SliceObject → SliceObject → PutObject',
            'steps': [
                {'action': 'PickupObject', 'command': 'pickup knife', 'object': 'knife'},
                {'action': 'SliceObject', 'command': 'slice the tomato', 'object': 'tomato'},
                {'action': 'SliceObject', 'command': 'slice the lettuce', 'object': 'lettuce'},
                {'action': 'PutObject', 'command': 'place knife', 'object': 'knife'}
            ],
            'description': 'Prepare a simple salad',
            'occurrences': 0,
            'hand_crafted': True
        },
        {
            'name': 'Cook_Breakfast',
            'pattern': 'OpenObject → PickupObject → CloseObject → CookObject',
            'steps': [
                {'action': 'OpenObject', 'command': 'open fridge', 'object': 'fridge'},
                {'action': 'PickupObject', 'command': 'pickup egg', 'object': 'egg'},
                {'action': 'CloseObject', 'command': 'close fridge', 'object': 'fridge'},
                {'action': 'CookObject', 'command': 'cook the egg', 'object': 'egg'}
            ],
            'description': 'Cook a simple breakfast',
            'occurrences': 0,
            'hand_crafted': True
        }
    ]
    
    for task in composite_tasks:
        task_library.append(task)
        print(f"\n{task['name']}: {task['pattern']}")
        print(f"   Description: {task['description']}")
        for step in task['steps']:
            print(f"      {step['command']}")
    
    # Save task library
    output = {
        'tasks': task_library,
        'statistics': {
            'total_tasks': len(task_library),
            'learned_from_youtube': len(common_patterns),
            'hand_crafted': len(composite_tasks),
            'unique_actions_used': len(set(
                step['action'] 
                for task in task_library 
                for step in task['steps']
            ))
        }
    }
    
    output_file = Path('data/processed/task_library.json')
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n" + "="*70)
    print(f"✅ Created library with {len(task_library)} tasks")
    print(f"   • {len(common_patterns)} learned from YouTube")
    print(f"   • {len(composite_tasks)} hand-crafted")
    print(f"💾 Saved to: {output_file}")
    print("="*70)
    
    return task_library

if __name__ == "__main__":
    task_library = create_task_library()
