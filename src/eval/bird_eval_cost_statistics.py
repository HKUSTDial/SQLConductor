import json
import os
import numpy as np
from typing import Dict, List, Any

"""
Generate token usage and execution-time statistics from answer.json.
Default output: answer_statistics.txt alongside the input file.

Usage:
    python -m src.eval.bird_eval_cost_statistics --predicted_file <path_to_answer.json> [--output_file <path_to_output_file>]
"""

def load_json(file_path: str) -> List[Dict[str, Any]]:
    """Load JSON data from file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_quartiles(data: List[float]) -> Dict[str, float]:
    """Calculate quartiles (Q1, Q2, Q3) and statistics for a list of numbers."""
    if not data:
        return {
            'min': 0, 'q1': 0, 'median': 0, 'q3': 0, 'max': 0,
            'mean': 0, 'std': 0, 'count': 0
        }
    
    data_array = np.array(data)
    return {
        'min': float(np.min(data_array)),
        'q1': float(np.percentile(data_array, 25)),
        'median': float(np.median(data_array)),
        'q3': float(np.percentile(data_array, 75)),
        'max': float(np.max(data_array)),
        'mean': float(np.mean(data_array)),
        'std': float(np.std(data_array)),
        'count': len(data)
    }


def extract_statistics(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract statistics from answer.json data."""
    stats = {
        'meta_agent': {
            'prompt_tokens': [],
            'completion_tokens': [],
            'total_tokens': []
        },
        'worker_usage': {
            'prompt_tokens': [],
            'completion_tokens': [],
            'total_tokens': []
        },
        'total': {
            'total_prompt_tokens': [],
            'total_completion_tokens': [],
            'total_tokens': [],
            'total_turns': [],
            'execution_time': []
        }
    }
    
    for item in data:
        if 'meta_agent_usage' in item and item['meta_agent_usage']:
            meta = item['meta_agent_usage']
            if 'prompt_tokens' in meta:
                stats['meta_agent']['prompt_tokens'].append(meta['prompt_tokens'])
            if 'completion_tokens' in meta:
                stats['meta_agent']['completion_tokens'].append(meta['completion_tokens'])
            if 'total_tokens' in meta:
                stats['meta_agent']['total_tokens'].append(meta['total_tokens'])
        
        if 'worker_usage' in item and item['worker_usage']:
            worker = item['worker_usage']
            if 'prompt_tokens' in worker:
                stats['worker_usage']['prompt_tokens'].append(worker['prompt_tokens'])
            if 'completion_tokens' in worker:
                stats['worker_usage']['completion_tokens'].append(worker['completion_tokens'])
            if 'total_tokens' in worker:
                stats['worker_usage']['total_tokens'].append(worker['total_tokens'])
        
        total_prompt = 0
        total_completion = 0
        has_usage_data = False
        
        if 'meta_agent_usage' in item and item['meta_agent_usage']:
            meta = item['meta_agent_usage']
            if 'prompt_tokens' in meta:
                total_prompt += meta['prompt_tokens']
                has_usage_data = True
            if 'completion_tokens' in meta:
                total_completion += meta['completion_tokens']
                has_usage_data = True
        
        if 'worker_usage' in item and item['worker_usage']:
            worker = item['worker_usage']
            if 'prompt_tokens' in worker:
                total_prompt += worker['prompt_tokens']
                has_usage_data = True
            if 'completion_tokens' in worker:
                total_completion += worker['completion_tokens']
                has_usage_data = True
        
        if has_usage_data:
            stats['total']['total_prompt_tokens'].append(total_prompt)
            stats['total']['total_completion_tokens'].append(total_completion)
        
        if 'total_tokens' in item:
            stats['total']['total_tokens'].append(item['total_tokens'])
        if 'total_turns' in item:
            stats['total']['total_turns'].append(item['total_turns'])
        if 'execution_time' in item:
            stats['total']['execution_time'].append(item['execution_time'])
    
    return stats


def format_statistics_section(title: str, stats_dict: Dict[str, List[float]]) -> List[str]:
    """Format a statistics table."""
    if not stats_dict:
        return []
    
    all_quartiles = {}
    for key, values in stats_dict.items():
        all_quartiles[key] = calculate_quartiles(values)
    
    headers = ['Metric', 'Count', 'Min', 'Q1', 'Median', 'Q3', 'Max', 'Mean', 'Std Dev']
    col_widths = [25, 10, 12, 12, 12, 12, 12, 12, 12]
    spacing = 2 
    
    total_width = sum(col_widths) + (spacing * (len(col_widths) - 1))
    
    heavy_line = "=" * total_width
    light_line = "-" * total_width
    
    lines = [
        "",
        "",
        title.center(total_width),
        heavy_line
    ]
    
    def format_row(data_list):
        formatted_cells = [str(data).center(w) for data, w in zip(data_list, col_widths)]
        return (" " * spacing).join(formatted_cells)

    lines.append(format_row(headers))
    lines.append(heavy_line)
    
    for key, quartiles in all_quartiles.items():
        metric_name = key.replace('_', ' ').title()
        row_data = [
            metric_name[:col_widths[0]-1], 
            f"{quartiles['count']}",
            f"{quartiles['min']:.2f}",
            f"{quartiles['q1']:.2f}",
            f"{quartiles['median']:.2f}",
            f"{quartiles['q3']:.2f}",
            f"{quartiles['max']:.2f}",
            f"{quartiles['mean']:.2f}",
            f"{quartiles['std']:.2f}"
        ]
        
        lines.append(format_row(row_data))
        lines.append(light_line)
    
    lines[-1] = heavy_line
    
    return lines

def generate_statistics_report(predicted_file: str, output_file: str = None):
    """Generate statistics report from answer.json file."""
    data = load_json(predicted_file)
    stats = extract_statistics(data)
    
    if output_file is None:
        predicted_dir = os.path.dirname(os.path.abspath(predicted_file))
        output_file = os.path.join(predicted_dir, 'answer_statistics.txt')
    
    lines = [
        '='*135,
        'STATISTICS REPORT',
        f'\nSource file: {predicted_file}',
        f'Total queries: {len(data)}',
        '='*135
    ]
    
    lines.extend(format_statistics_section(
        'Meta Agent Usage Statistics',
        stats['meta_agent']
    ))
    
    lines.extend(format_statistics_section(
        'Worker Usage Statistics',
        stats['worker_usage']
    ))
    
    lines.extend(format_statistics_section(
        'Total Statistics',
        stats['total']
    ))

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f'\nCost statistics report saved to: {output_file}')


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate statistics report from answer.json')
    parser.add_argument('--predicted_file', type=str, required=True,
                        help='Path to answer.json file')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output statistics file path (default: answer_statistics.txt in same directory)')
    
    args = parser.parse_args()
    generate_statistics_report(args.predicted_file, args.output_file)
