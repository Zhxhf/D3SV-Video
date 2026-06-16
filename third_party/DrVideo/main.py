import os
import cv2
import ast
import json
import logging
import copy
from pathlib import Path
from tqdm import tqdm
from pprint import pprint
from typing import Dict, Any, List

# Assuming these are your custom modules
from util import parse_args, makedir, load_json, save_json
from eval import eval_qa_egoschema, eval_sum
from dataset import get_dataset
from prompts import PromptFactory
from model import get_model
from models.blip2_model import ImageCaptioner

# LangChain imports
from langchain.document_loaders import JSONLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# --- Constants ---
ANSWER_MAPPING = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
PROMPT_TEMPLATES = {
    'A': "[INST] <image>\nWhat is shown in this image? [/INST]",
    'B': "[INST] <image>\n{question} [/INST]",
    'C': "[INST] <image>\nPlease describe this image in no more than 100 words and do not miss key visual elements, such as objects, humans, interactions, actions, and scenes. [/INST]"
}

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def choice_to_int(choice: str) -> int:
    """Converts an alphabetical choice ('A', 'B', etc.) to a zero-based integer."""
    return ANSWER_MAPPING.get(choice, -1)


def safe_llm_output_parse(output_str: str) -> Any:
    """Safely parses a stringified Python literal from an LLM's output."""
    try:
        return ast.literal_eval(output_str)
    except (ValueError, SyntaxError, TypeError) as e:
        logging.error(f"Failed to parse LLM output: {output_str}. Error: {e}")
        return None


def get_relevant_frames(json_file_path: Path, question: str, k: int = 20) -> Dict[str, str]:
    """Retrieves the most relevant frame captions from a JSON database using a vector store."""
    loader = JSONLoader(file_path=str(json_file_path), jq_schema='.messages[].content')
    docs = loader.load()
    
    vectorstore = Chroma.from_documents(documents=docs, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    
    relevant_docs = retriever.get_relevant_documents(question)
    
    frame_dict = {}
    for doc in relevant_docs:
        try:
            frame_index, frame_content = doc.page_content.split(" ", 1)
            frame_dict[frame_index] = frame_content
        except ValueError:
            logging.warning(f"Could not split document content into frame_index and content: {doc.page_content}")
            
    return frame_dict


def generate_captions(video_path: str, frames_to_caption: List[Dict], question: str, image_captioner: ImageCaptioner) -> Dict[str, str]:
    """
    Generates captions for a list of specified frames from a video.
    `frames_to_caption` is a list of dicts, e.g., [{'frame': '5', 'type': 'A'}, {'frame': '12', 'type': 'B'}].
    """
    captions = {}
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Could not open video file: {video_path}")
        return {}
        
    fps = cap.get(cv2.CAP_PROP_FPS)

    for frame_info in frames_to_caption:
        try:
            key_frame_sec = int(frame_info["frame"])
            frame_type = frame_info.get("type", 'C') # Default to dense description
            
            # Position at the middle of the second for robustness
            frame_idx = int((key_frame_sec - 0.5) * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                prompt_template = PROMPT_TEMPLATES.get(frame_type, PROMPT_TEMPLATES['C'])
                text_prompt = prompt_template.format(question=question)
                
                caption_output = image_captioner.image_caption(frame_rgb, text_prompt, frame_type)
                # Assumes the output format is "[/INST] caption_text"
                caption = caption_output.split("[/INST]", 1)[-1].strip()
                
                captions[str(key_frame_sec)] = caption
            else:
                logging.warning(f"Could not read frame at second {key_frame_sec} from {video_path}")
        except (KeyError, ValueError) as e:
            logging.error(f"Error processing frame_info {frame_info}: {e}")

    cap.release()
    return captions


def process_item(item: Dict, models: Dict, prompters: Dict, image_captioner: ImageCaptioner, args: Any) -> Dict:
    """
    Processes a single data item (video question) through the entire pipeline.
    """
    ukey_name = 'quid' if 'quid' in item else 'uid'
    ukey = item[ukey_name]
    json_file_path = Path('./data/egoschema/database/') / f"{ukey}.json"

    # 1. Initial Retrieval
    raw_doc = load_json(json_file_path)
    all_captions = {str(doc["time"]): doc["content"].split(" ", 1)[1] for doc in raw_doc["messages"]}
    
    relevant_frames_dict = get_relevant_frames(json_file_path, item['question'], k=20)
    key_frame_indices = sorted([int(k) for k in relevant_frames_dict.keys()])
    
    # 2. Augment initial captions with question-specific context
    frames_for_initial_captioning = [{'frame': str(idx), 'type': 'B'} for idx in key_frame_indices]
    initial_new_captions = generate_captions(item['video_path'], frames_for_initial_captioning, item['question'], image_captioner)
    for frame_idx, caption in initial_new_captions.items():
        all_captions[frame_idx] += f" [Question-specific context: {caption}]"

    # 3. Iterative Feedback Loop
    max_turns = 2
    type_A_frames, type_B_frames = [], key_frame_indices.copy()
    gpt_feedback_history = ""
    
    for turn in range(max_turns):
        logging.info(f"--- Ukey: {ukey}, Turn: {turn + 1}/{max_turns} ---")
        
        current_context_str = "\n".join([f"frame {k}: {v}" for k, v in sorted(all_captions.items())])
        
        # Judge if the current context is sufficient
        prompt_judge = prompters['judge'].fill(**item, captions=current_context_str, question_context=item['question'], gpt_prompt=gpt_feedback_history)
        pred_judge_str, _ = models['judge'].forward(prompters['judge'].head, prompt_judge)
        pred_judge = safe_llm_output_parse(pred_judge_str)
        
        logging.info(f"Judge output: {pred_judge}")
        
        if pred_judge and pred_judge.get('confidence') == '1':
            logging.info("Judge is confident. Breaking feedback loop.")
            break
        
        # Find new frames to analyze if context is insufficient
        judge_explanation = f"In round {turn + 1}, the reasoning was: {pred_judge.get('explanation', 'N/A')}\n"
        prompt_find = prompters['find'].fill(**item, captions=current_context_str, question_context=item['question'], explanation=judge_explanation, type_A=str(type_A_frames), type_B=str(type_B_frames))
        pred_find_str, _ = models['find'].forward(prompters['find'].head, prompt_find)
        frames_to_add = safe_llm_output_parse(pred_find_str)
        
        logging.info(f"Find output: {frames_to_add}")
        
        if not frames_to_add:
            logging.warning("Find model returned no valid frames to add. Breaking loop.")
            break
            
        # Generate new captions and update context
        new_captions = generate_captions(item['video_path'], frames_to_add, item['question'], image_captioner)
        all_captions.update(new_captions)
        
        # Update feedback history for the next turn's prompt
        new_info_str = "\n".join([f"frame {k}: {v}" for k, v in new_captions.items()])
        gpt_feedback_history += f"In round {turn + 1}, we added the following information:\n{new_info_str}\n"

        for f_info in frames_to_add:
            if f_info['type'] == 'A': type_A_frames.append(int(f_info['frame']))

    # 4. Final Reasoning and Prediction
    final_context_str = "\n".join([f"frame {k}: {v}" for k, v in sorted(all_captions.items())])
    prompt_reasoning = prompters['reasoning'].fill(**item, context=final_context_str, question_text=item['question'], option1=item['optionA'], option2=item['optionB'], option3=item['optionC'], option4=item['optionD'], option5=item['optionE'])
    pred_reasoning_str, info = models['reasoning'].forward(prompters['reasoning'].head, prompt_reasoning)
    pred_reasoning = safe_llm_output_parse(pred_reasoning_str)
    
    # 5. Format Output
    result = copy.deepcopy(item)
    result.update({
        'rag_prompt_template': prompters['judge'].get_template_str(),
        'reasoning_prompt_template': prompters['reasoning'].get_template_str(),
        'response': pred_reasoning.get('final_answer', 'ERROR') if pred_reasoning else 'ERROR',
        'pred': choice_to_int(pred_reasoning.get('final_answer', '')) if pred_reasoning else -1
    })
    
    if args.save_info:
        result['info'] = {k: v for k, v in info.items() if k != 'response'}
        
    return result

def main():
    """Main execution function."""
    args = parse_args()
    pprint(args)
    os.environ["OPENAI_API_KEY"] = args.openai_api_key

    # --- Setup ---
    output_dir = Path(args.output_base_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_filename

    # Initialize models and prompters ONCE
    logging.info("Initializing models and prompters...")
    image_captioner = ImageCaptioner(model_name=args.captioner_base_model, device=args.image_captioner_device)
    
    prompters = {
        'judge': PromptFactory().get('judge'),
        'find': PromptFactory().get('find'),
        'reasoning': PromptFactory().get('llm_reasoning')
    }
    
    models = {
        'judge': get_model(args),
        'find': get_model(args),
        'reasoning': get_model(args)
    }
    models['judge'].set_post_process_fn(prompters['judge'].post_process_fn)
    models['find'].set_post_process_fn(prompters['find'].post_process_fn)
    models['reasoning'].set_post_process_fn(prompters['reasoning'].post_process_fn)
    
    # --- Data Loading & Resuming ---
    processed = {}
    if not args.start_from_scratch and output_path.exists():
        processed_data = load_json(output_path)
        processed = processed_data.get('data', processed_data)
        logging.info(f"Resuming run. Loaded {len(processed)} processed items.")

    quids_to_exclude = set(processed.keys())
    dataset = get_dataset(args, quids_to_exclude=quids_to_exclude, num_examples_to_run=args.num_examples_to_run)
    
    # --- Main Loop ---
    correct_answers = 0
    total_processed_count = len(processed)
    
    for i, item in enumerate(tqdm(dataset, desc="Processing Items")):
        ukey_name = 'quid' if 'quid' in item else 'uid'
        ukey = item[ukey_name]

        try:
            result = process_item(item, models, prompters, image_captioner, args)
            processed[ukey] = result
            
            if result['pred'] == result['truth']:
                correct_answers += 1
            
            total_processed_count += 1
            accuracy = (correct_answers / (i + 1)) * 100
            logging.info(f"✅ Correct: {result['pred'] == result['truth']} | Prediction: {result['response']} | Truth: {chr(ord('A') + result['truth'])}")
            logging.info(f"Accuracy so far: {accuracy:.2f}% ({correct_answers}/{i + 1})")

        except Exception as e:
            logging.critical(f"FATAL: An unexpected error occurred while processing {ukey}: {e}", exc_info=True)
            # Optionally, save a placeholder for the failed item
            processed[ukey] = {**item, 'response': 'PROCESSING_ERROR', 'pred': -1}

        if (i + 1) % args.save_every == 0:
            save_json(processed, output_path)

    # --- Finalization & Evaluation ---
    if len(args.backup_pred_path) > 0:
        backup = load_json(args.backup_pred_path)
        backup_data = backup.get('data', backup)
        for uid, data in processed.items():
            if data['pred'] == -1 and uid in backup_data:
                data['pred'] = backup_data[uid]['pred']
    
    if not args.disable_eval:
        if args.task == 'qa' and args.dataset == 'egoschema':
            processed = eval_qa_egoschema(processed)
        elif args.task == 'sum':
            processed, sum_data = eval_sum(processed)
            save_json(sum_data, output_path.with_name(f'{output_path.stem}_data.json'))

    save_json(processed, output_path)
    logging.info(f"Processing complete. Final results saved to {output_path}")

if __name__ == '__main__':
    main()