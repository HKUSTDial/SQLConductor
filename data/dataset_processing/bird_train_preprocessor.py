#!/usr/bin/env python3
"""Prepare question IDs and filter databases for BIRD training."""

import json

input_file = "data/bird/train/train.json"
output_file = "data/bird/train/train_processed.json"

BLOCKED_DATABASES = [
    "codebase_comments",
    "donor", 
    "language_corpus",
    "movie_platform",
    "music_platform_2",
    "retails",
    "authors",
    "bike_share_1",
    "menu",
    "world_development_indicators"
]


def load_data(file_path: str) -> list:
    """Load JSON data from file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} records")
    return data


def fix_question_ids(data: list) -> int:
    """Populate question IDs from existing IDs or original record positions."""
    fixed_count = 0
    for index, record in enumerate(data):
        if 'question_id' not in record:
            record['question_id'] = record.pop('id', index)
            fixed_count += 1
    
    print(f"Prepared question IDs for {fixed_count} records")
    return fixed_count


def filter_blocked_databases(data: list, blocked_databases: list) -> tuple:
    """Filter out records from blocked databases"""
    filtered_data = []
    blocked_count = 0
    
    for record in data:
        if 'db_id' in record and record['db_id'] in blocked_databases:
            blocked_count += 1
        else:
            filtered_data.append(record)
    
    print(f"Filtered out {blocked_count} records from blocked databases")
    print(f"Remaining records: {len(filtered_data)}")
    
    return filtered_data, blocked_count


def save_data(data: list, file_path: str) -> None:
    """Save data to JSON file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved to: {file_path}")


def process_bird_train():
    data = load_data(input_file)
    
    fix_question_ids(data)
    
    filtered_data, blocked_count = filter_blocked_databases(data, BLOCKED_DATABASES)
    
    save_data(filtered_data, output_file)


if __name__ == "__main__":
    try:
        process_bird_train()
        print("Processing completed!")
    except Exception as e:
        print(f"Error: {e}")
