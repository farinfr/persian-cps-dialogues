#!/usr/bin/env python3
"""
Add Kind Property to LLM Generated Dialogues
============================================

This script adds a 'kind' property with value 3 to all JSON files in llm_generated_dialogues.
"""

import json
import os
from typing import Dict, Any

def add_kind_to_file(file_path: str, kind: int = 3):
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
    print("Adding Kind Property to LLM Generated Dialogues")
    print("=" * 60)
    
    llm_dir = "llm_generated_dialogues"
    
    if not os.path.exists(llm_dir):
        print(f"Error: Directory {llm_dir} not found")
        return
    
    # Process main file
    main_file = os.path.join(llm_dir, "llm_generated_dialogues.json")
    if os.path.exists(main_file):
        print(f"\nProcessing main file: {main_file}")
        with open(main_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Add kind to all dialogues
        if 'generated_dialogues' in data:
            for dialogue in data['generated_dialogues']:
                dialogue['kind'] = 3
        
        with open(main_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  ✅ Updated main file with {len(data.get('generated_dialogues', []))} dialogues")
    
    # Process individual files
    print(f"\nProcessing individual files...")
    files_processed = 0
    
    for filename in os.listdir(llm_dir):
        if not filename.endswith('.json'):
            continue
        
        # Skip main file (already processed)
        if filename == 'llm_generated_dialogues.json':
            continue
        
        file_path = os.path.join(llm_dir, filename)
        add_kind_to_file(file_path, kind=3)
        files_processed += 1
    
    print(f"\n✅ Process completed!")
    print(f"  Total individual files processed: {files_processed}")

if __name__ == "__main__":
    main()


