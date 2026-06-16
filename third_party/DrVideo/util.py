import pickle
import json
from pathlib import Path
import argparse
import pandas as pd
from pprint import pprint


def load_pkl(fn):
    with open(fn, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pkl(data, fn):
    with open(fn, 'wb') as f:
        pickle.dump(data, f)

def load_json(fn):
    with open(fn, 'r') as f:
        data = json.load(f)
    return data

def save_json(data, fn, indent=4):
    with open(fn, 'w') as f:
        json.dump(data, f, indent=indent)

def makedir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser("")

    # data
    parser.add_argument("--dataset", default='egoschema', type=str)  # 'egoschema', 'nextqa', 'nextgqa', 'intentqa'
    parser.add_argument("--data_path", default='data/egoschema/lavila_subset_add_time.json', type=str) 
    parser.add_argument("--anno_path", default='data/egoschema/subset_anno.json', type=str)  
    parser.add_argument("--duration_path", default='data/egoschema/duration.json', type=str) 
    parser.add_argument("--fps", default=1.0, type=float) 
    parser.add_argument("--num_examples_to_run", default=-1, type=int)
    ## backup pred
    parser.add_argument("--backup_pred_path", default="", type=str)
    ## fewshot
    parser.add_argument("--fewshot_example_path", default="", type=str) 
    ## nextgqa
    parser.add_argument("--nextgqa_gt_ground_path", default="", type=str)
    parser.add_argument("--nextgqa_pred_qa_path", default="", type=str)

    # output
    parser.add_argument("--output_base_path", required=True, type=str)  
    parser.add_argument("--output_filename", required=True, type=str)  

    # prompting
    parser.add_argument("--model", default="gpt-4-turbo", type=str)
    #parser.add_argument("--api_key", default="", type=str)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--prompt_type", default="qa_standard", type=str)
    parser.add_argument("--task", default="qa", type=str)  # sum, qa, gqa
    ## sum
    parser.add_argument("--num_words_in_sum", default=50, type=int)  

    # other
    parser.add_argument("--disable_eval", action='store_true')
    parser.add_argument("--start_from_scratch", action='store_true')
    parser.add_argument("--save_info", action='store_true')
    parser.add_argument("--save_every", default=10, type=int)

    #other from vlog
    parser.add_argument('--video_path', default='examples/travel_in_roman.mp4')
    parser.add_argument('--alpha', default=20, type=int, help='Determine the maximum segment number for KTS algorithm, the larger the value, the fewer segments.')
    parser.add_argument('--beta', default=1, type=int, help='The smallest time gap between successive clips, in seconds.')
    parser.add_argument('--data_dir', default='./data/egoschema/database-log', type=str, help='Directory for saving videos and logs.')
    parser.add_argument('--tmp_dir', default='./data/egoschema/database-pkl', type=str, help='Directory for saving intermediate files.')
    
    # * Models settings *
    parser.add_argument('--openai_api_key', default='', type=str, help='OpenAI API key')
    parser.add_argument('--image_caption', action='store_true', dest='image_caption', default=True, help='Set this flag to True if you want to use BLIP Image Caption')
    parser.add_argument('--dense_caption', action='store_true', dest='dense_caption', default=True, help='Set this flag to True if you want to use Dense Caption')
    parser.add_argument('--feature_extractor', default='openai/clip-vit-base-patch32', help='Select the feature extractor model for video segmentation')
    parser.add_argument('--feature_extractor_device', choices=['cuda', 'cpu'], default='cuda', help='Select the device: cuda or cpu')
    parser.add_argument('--image_captioner', choices=['blip2-opt', 'blip2-flan-t5', 'blip'], dest='captioner_base_model', default='blip2-flan-t5', help='blip2 requires 15G GPU memory, blip requires 6G GPU memory')
    parser.add_argument('--image_captioner_device', choices=['cuda', 'cpu'], default='cuda', help='Select the device: cuda or cpu, gpu memory larger than 14G is recommended')
    parser.add_argument('--dense_captioner_device', choices=['cuda', 'cpu'], default='cuda', help='Select the device: cuda or cpu, < 6G GPU is not recommended>')
    parser.add_argument('--audio_translator', default='large')
    parser.add_argument('--audio_translator_device', choices=['cuda', 'cpu'], default='cuda')


    return parser.parse_args()


def build_fewshot_examples(qa_path, data_path):
    if len(qa_path) == 0 or len(data_path) == 0:
        return None
    qa = load_json(qa_path)
    data = load_json(data_path)  # uid --> str or list 
    examplars = []
    int_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
    for i, (uid, examplar) in enumerate(qa.items()):
        description = data[uid]
        if isinstance(description, list):
            description = '. '.join(description)
        examplars.append(f"Examplar {i}.\n Descriptions: {description}.\n Question: {examplar['question']}\n A: {examplar['0']}\n B: {examplar['1']}\n C: {examplar['2']}\n D: {examplar['3']}\n E: {examplar['4']}\n Answer: {int_to_letter[examplar['truth']]}.")
    examplars = '\n\n'.join(examplars)
    return examplars
    
    
    