import sys
import json
import argparse
import sqlite3
import multiprocessing as mp
import os
from contextlib import closing
from pathlib import Path
from func_timeout import func_timeout, FunctionTimedOut
from tqdm import tqdm

from .bird_eval_cost_statistics import generate_statistics_report


"""
Usage:
    python -m src.eval.bird_eval \
    --predicted_file results/inference/ans/sequence/answer.json \
    --ground_truth_file data/bird/dev/dev.json \
    --db_root_path data/bird/dev/dev_databases \
    --num_cpus 16
"""

def load_json(dir):
    with open(dir, 'r', encoding='utf-8') as j:
        contents = json.loads(j.read())
    return contents

def execute_sql_single(sql, db_path):
    with closing(sqlite3.connect(Path(db_path).resolve().as_uri() + '?mode=ro', uri=True)) as conn:
        cursor = conn.execute(sql)
        if cursor.description is None:
            raise ValueError('SQL query returned no result columns.')
        return cursor.fetchall()

def execute_model(predicted_sql, ground_truth, db_place, idx, meta_time_out):
    res = 0
    status = ''
    predicted_res_list = None
    
    # Skip tasks whose reference SQL fails or times out.
    try:
        ground_truth_res = func_timeout(meta_time_out, execute_sql_single, args=(ground_truth, db_place))
    except FunctionTimedOut:
        return {'sql_idx': idx, 'res': None, 'status': 'gold_timeout', 'predicted_empty': False}
    except Exception:
        return {'sql_idx': idx, 'res': None, 'status': 'gold_error', 'predicted_empty': False}

    try:
        predicted_res_list = func_timeout(meta_time_out, execute_sql_single, args=(predicted_sql, db_place))
        if set(predicted_res_list) == set(ground_truth_res):
            res = 1
            status = 'correct'
        else:
            res = 0
            status = 'incorrect'
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        res = 0
        status = 'timeout'
    except Exception:
        res = 0
        status = 'error'
    
    predicted_empty = predicted_res_list is not None and len(predicted_res_list) == 0
    return {'sql_idx': idx, 'res': res, 'status': status, 'predicted_empty': predicted_empty}

def execute_model_wrapper(args):
    return execute_model(*args)

def load_and_pair_sqls(predicted_file, ground_truth_file, db_root_path):
    gt_data = load_json(ground_truth_file)
    pred_data = load_json(predicted_file)
    pred_map = {item['question_id']: item for item in pred_data}

    query_pairs = []
    db_paths = []

    for gt_item in gt_data:
        question_id = gt_item['question_id']
        gt_sql = gt_item['SQL']
        db_id = gt_item['db_id']
        
        pred_item = pred_map.get(question_id)
        
        if pred_item and pred_item.get('final_sql'):
            pred_sql = pred_item['final_sql']
        else:
            pred_sql = None
            
        query_pairs.append((pred_sql, gt_sql))
        db_path = f"{db_root_path}/{db_id}/{db_id}.sqlite"
        db_paths.append(db_path)
        
    return query_pairs, db_paths


def run_sqls_parallel(sqls, db_places, num_cpus=1, meta_time_out=20.0):
    pool = mp.Pool(processes=num_cpus)
    
    args_list = [
        (sql_pair[0], sql_pair[1], db_places[i], i, meta_time_out)
        for i, sql_pair in enumerate(sqls)
    ]

    results = []
    with tqdm(total=len(sqls)) as pbar:
        for result in pool.imap_unordered(execute_model_wrapper, args_list):
            if result:
                results.append(result)
            pbar.update()

    pool.close()
    pool.join()
    return results

def sort_results(list_of_dicts):
  return sorted(list_of_dicts, key=lambda x: x['sql_idx'])

def compute_acc_by_diff(exec_results,diff_json_path):
    valid_exec_results = [res for res in exec_results if res['res'] is not None]
    num_queries = len(valid_exec_results)
    results = [res['res'] for res in valid_exec_results]
    contents = load_json(diff_json_path)
    simple_results, moderate_results, challenging_results = [], [], []

    idx_to_res = {res['sql_idx']: res for res in valid_exec_results}

    for i, content in enumerate(contents):
        if i in idx_to_res:
            res_item = idx_to_res[i]
            if content['difficulty'] == 'simple':
                simple_results.append(res_item)
            elif content['difficulty'] == 'moderate':
                moderate_results.append(res_item)
            elif content['difficulty'] == 'challenging':
                challenging_results.append(res_item)

    simple_correct = sum([res['res'] for res in simple_results])
    moderate_correct = sum([res['res'] for res in moderate_results])
    challenging_correct = sum([res['res'] for res in challenging_results])
    all_correct = sum(results)

    simple_acc = simple_correct / len(simple_results) if len(simple_results) > 0 else 0
    moderate_acc = moderate_correct / len(moderate_results) if len(moderate_results) > 0 else 0
    challenging_acc = challenging_correct / len(challenging_results) if len(challenging_results) > 0 else 0
    all_acc = all_correct / num_queries if num_queries > 0 else 0
    count_lists = [len(simple_results), len(moderate_results), len(challenging_results), num_queries]
    correct_counts = [simple_correct, moderate_correct, challenging_correct, all_correct]
    return simple_acc * 100, moderate_acc * 100, challenging_acc * 100, all_acc * 100, count_lists, correct_counts


def print_data(score_lists,count_lists, correct_counts, output_file=None):
    levels = ['simple', 'moderate', 'challenging', 'total']
    lines = [
        "{:20} {:20} {:20} {:20} {:20}".format("", *levels),
        '--------------------------------------------------------------------------------------------------',
        "{:20} {:<20} {:<20} {:<20} {:<20}".format('count', *count_lists),
        '--------------------------------------       ACCURACY        -------------------------------------',
        "{:20} {:<20} {:<20} {:<20} {:<20}".format('correct', *correct_counts),
        "{:20} {:<20.2f} {:<20.2f} {:<20.2f} {:<20.2f}".format('accuracy (%)', *score_lists)
    ]
    for line in lines:
        print(line)
        if output_file:
            output_file.write(line + '\n')


def print_detailed_report(exec_results, output_file=None):
    valid_exec_results = [res for res in exec_results if res['res'] is not None]
    num_queries = len(valid_exec_results)
    
    gold_error_count = sum(1 for r in exec_results if r['status'] == 'gold_error')
    gold_timeout_count = sum(1 for r in exec_results if r['status'] == 'gold_timeout')
    skipped_count = gold_error_count + gold_timeout_count
    
    correct_count = sum(1 for r in valid_exec_results if r['status'] == 'correct')
    incorrect_count = sum(1 for r in valid_exec_results if r['status'] == 'incorrect')
    error_count = sum(1 for r in valid_exec_results if r['status'] == 'error')
    timeout_count = sum(1 for r in valid_exec_results if r['status'] == 'timeout')
    empty_res_count = sum(1 for r in valid_exec_results if r['predicted_empty'])

    lines = [
        '--------------------------------------    DETAILED REPORT    -------------------------------------',
        f"{'Total queries:':<20} {len(exec_results):<20}",
        f"{'Skipped (Gold Err):':<20} {gold_error_count:<20}",
        f"{'Skipped (Gold TO):':<20} {gold_timeout_count:<20}",
        f"{'Correct:':<20} {correct_count:<20}",
        f"{'Incorrect:':<20} {incorrect_count:<20}",
        f"{'Execution errors:':<20} {error_count:<20}",
        f"{'Timeouts:':<20} {timeout_count:<20}",
        f"{'Empty results:':<20} {empty_res_count:<20}",
        '=================================================================================================='
    ]
    for line in lines:
        print(line)
        if output_file:
            output_file.write(line + '\n')


def write_back_results(exec_results, gt_file, pred_file):
    """Write evaluation results back to the prediction file."""
    gt_data = load_json(gt_file)
    pred_data = load_json(pred_file)

    # exec_results are sorted according to gt_data
    results_map = {}
    for i, gt_item in enumerate(gt_data):
        question_id = gt_item['question_id']
        result = exec_results[i]
        results_map[question_id] = {
            'eval_correct': bool(result['res']),
            'eval_status': result['status'],
            'eval_predicted_empty': result['predicted_empty']
        }

    for pred_item in pred_data:
        question_id = pred_item.get('question_id')
        if question_id is not None and question_id in results_map:
            pred_item.update(results_map[question_id])

    with open(pred_file, 'w', encoding='utf-8') as f:
        json.dump(pred_data, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--predicted_file', type=str, required=True, default='')
    args_parser.add_argument('--ground_truth_file', type=str, required=True, default='')
    args_parser.add_argument('--db_root_path', type=str, required=True, default='')
    args_parser.add_argument('--num_cpus', type=int, default=1)
    args_parser.add_argument('--meta_time_out', type=float, default=20.0)
    args = args_parser.parse_args()
    exec_result = []

    query_pairs, db_paths = load_and_pair_sqls(args.predicted_file, args.ground_truth_file, args.db_root_path)
    exec_result = run_sqls_parallel(query_pairs, db_places=db_paths, num_cpus=args.num_cpus, meta_time_out=args.meta_time_out)
    exec_result = sort_results(exec_result)
    
    predicted_dir = os.path.dirname(os.path.abspath(args.predicted_file))
    output_txt_path = os.path.join(predicted_dir, 'answer_ex.txt')
    
    with open(output_txt_path, 'w', encoding='utf-8') as output_file:
        info_lines = [
            f'Examined file dir: {args.predicted_file}',
            f'Examined database dir: {args.db_root_path}',
            f'Ground truth file: {args.ground_truth_file}',
            '=================================================================================================='
        ]
        for line in info_lines:
            print(line)
            output_file.write(line + '\n')
        
        simple_acc, moderate_acc, challenging_acc, acc, count_lists, correct_counts = \
            compute_acc_by_diff(exec_result, args.ground_truth_file)
        score_lists = [simple_acc, moderate_acc, challenging_acc, acc]
        print_data(score_lists, count_lists, correct_counts, output_file)
        print_detailed_report(exec_result, output_file)

    write_back_results(exec_result, args.ground_truth_file, args.predicted_file)

    print(f"Evaluation results have been written back to {args.predicted_file}")
    print(f"Evaluation results have been saved to {output_txt_path}")
    
    generate_statistics_report(args.predicted_file)
