#!/usr/bin/env python3
"""
Add Kind Property to Dual Method Dialogues
==========================================

This script adds a 'kind' property to all JSON files in dual_method_dialogues:
- kind = 1 for highest metric method
- kind = 2 for random method
"""

import json
import os
from typing import Dict, Any

def add_kind_to_file(file_path: str, kind: int):
    """Add kind property to a JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Add kind property at the top level
    data['kind'] = kind
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"  ✅ Updated {os.path.basename(file_path)} with kind={kind}")

def main():
    """Main function"""
    print("Adding Kind Property to Dual Method Dialogues")
    print("=" * 60)
    
    dual_method_dir = "dual_method_dialogues"
    
    if not os.path.exists(dual_method_dir):
        print(f"Error: Directory {dual_method_dir} not found")
        return
    
    # Process main file
    main_file = os.path.join(dual_method_dir, "llm_rewritten_dialogues.json")
    if os.path.exists(main_file):
        print(f"\nProcessing main file: {main_file}")
        with open(main_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Add kind to random method dialogues
        if 'random_method_rewritten' in data and 'dialogues' in data['random_method_rewritten']:
            for dialogue in data['random_method_rewritten']['dialogues']:
                dialogue['kind'] = 2
        
        # Add kind to highest metric method dialogues
        if 'highest_metric_method_rewritten' in data and 'dialogues' in data['highest_metric_method_rewritten']:
            for dialogue in data['highest_metric_method_rewritten']['dialogues']:
                dialogue['kind'] = 1
        
        with open(main_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  ✅ Updated main file")
    
    # Process template file
    template_file = os.path.join(dual_method_dir, "template_dialogues.json")
    if os.path.exists(template_file):
        print(f"\nProcessing template file: {template_file}")
        with open(template_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Add kind to random method dialogues
        if 'random_method' in data and 'dialogues' in data['random_method']:
            for dialogue in data['random_method']['dialogues']:
                dialogue['kind'] = 2
        
        # Add kind to highest metric method dialogues
        if 'highest_metric_method' in data and 'dialogues' in data['highest_metric_method']:
            for dialogue in data['highest_metric_method']['dialogues']:
                dialogue['kind'] = 1
        
        with open(template_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  ✅ Updated template file")
    
    # Process individual files
    print(f"\nProcessing individual files...")
    files_processed = 0
    
    for filename in os.listdir(dual_method_dir):
        if not filename.endswith('.json'):
            continue
        
        # Skip main files (already processed)
        if filename in ['llm_rewritten_dialogues.json', 'template_dialogues.json']:
            continue
        
        file_path = os.path.join(dual_method_dir, filename)
        
        # Determine kind based on filename
        if filename.startswith('highest_metric_rewritten_'):
            kind = 1
        elif filename.startswith('random_rewritten_'):
            kind = 2
        else:
            print(f"  ⚠️  Skipping {filename} (unknown type)")
            continue
        
        add_kind_to_file(file_path, kind)
        files_processed += 1
    
    print(f"\n✅ Process completed!")
    print(f"  Total individual files processed: {files_processed}")

if __name__ == "__main__":
    main()


