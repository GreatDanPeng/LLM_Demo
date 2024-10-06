import argparse
import os
import sys
import random
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from transformers import StoppingCriteriaList

sys.path.append(os.path.join(os.path.dirname(__file__), 'MiniGPT-4'))
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, StoppingCriteriaSub
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'dataLoader'))
from MME_dataloader import MMEdataLoader

# Reduce GPU memory
import gc
gc.collect()
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.9, max_split_size_mb:128"

class MiniGPT4Inference:
    def __init__(self, CONV_VISION = CONV_VISION_Vicuna0, config = None, args = None):
        self.CONV_VISION = CONV_VISION_Vicuna0
        self.model_config = config.model_cfg
        self.model_config.device_8bit = args.gpu_id
        self.model_cls = registry.get_model_class(self.model_config.arch)
        self.model = self.model_cls.from_config(self.model_config)
        self.device = torch.device("cuda")
        self.model.to(self.device)
        self.vis_processor_cfg = config.datasets_cfg.cc_sbu_align.vis_processor.train
        self.vis_processor = registry.get_processor_class(self.vis_processor_cfg.name).from_config(self.vis_processor_cfg)
        self.chat = Chat(self.model, self.vis_processor, self.device)

    def generate_answer(self, question):
        inputs = self.tokenizer(question, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer

def parse_args():
    parser = argparse.ArgumentParser(description="MiniGPT-4 Inference")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

def process_dataset(file_path, category):
    print(f"Processing dataset: {file_path}")
    mme_loader = MMEdataLoader(file_path)
    mme_dataset = mme_loader.load_data()
    print("Finished Loading!S")
    results_by_category[category] = {
        'results': [],
        'start_time': time.time()
    }

    for i in range(0, len(mme_dataset['question_id'])):
        question_id = mme_dataset['question_id'][i]
        image = mme_dataset['image'][i]
        question = mme_dataset['question'][i]
        gt = mme_dataset['answer'][i]
        category = mme_dataset['category'][i]
        print(f"Loaded image: {image}")
        print(f"Loaded question: {question}")
        print(f"Loaded category: {category}")
        print("Initializing conversation")
        conv = Mini.CONV_VISION.copy()
        img_list = []
        Mini.chat.upload_img(image, conv, img_list)
        Mini.chat.encode_img(img_list)
        Mini.chat.ask(f"{question}", conv)
        answer = Mini.chat.answer(conv=conv,
                                  img_list=img_list,
                                  num_beams=1,
                                  temperature=1.0,
                                  max_new_tokens=10, 
                                  max_length=20000)[0]
        
        if "no" in answer.lower():
            answer = "no"
        elif "yes" in answer.lower():
            answer = "yes"
        else:
            answer = "other"
        print(f"Answer: {answer}")
        
        results_by_category[category]['results'].append({
            'question_id': question_id, 
            'question': question,  
            'gt': gt,
            'answer': answer
        })

        # Sort the results by question_id
        results_by_category[category]['results'] = sorted(
        results_by_category[category]['results'], 
        key=lambda x: x['question_id']
        )

        # Clear GPU cache to avoid memory issues
        torch.cuda.empty_cache()
        gc.collect()

    save_results_by_category(category)
    save_gt_by_category(category)

    end_time = time.time()
    elapsed_time = end_time - results_by_category[category]['start_time']
    print(f"Category '{category}' processed in {elapsed_time:.2f} seconds")
    
def save_results_by_category(category):
    file_path = f"Output/{category}.txt"
    with open(file_path, 'w', encoding='utf-8') as file:
        for result in results_by_category[category]['results']:
            file.write(f"{result['question_id']}\t{result['question']}\t{result['gt']}\t{result['answer']}\n")

def save_gt_by_category(category):
    file_path = f"Your_Results/{category}.txt"
    with open(file_path, 'w', encoding='utf-8') as file:
        for result in results_by_category[category]['results']:
            file.write(f"{result['question_id']}\t{result['question']}\t{result['gt']}\n")

if __name__ == "__main__":
    total_start_time = time.time()
    args = parse_args()
    cfg = Config(args)
    setup_seeds(cfg)

    print('Initializing Chat')
    Mini = MiniGPT4Inference(config = cfg, args = args)
    print('Initialization Finished')

    ### Apply MiniGPT to the MME dataset
    # MME data preprocess
    MME_path = "../MME/"
    results_by_category = {}

    for file_name in os.listdir(MME_path):
        if file_name.endswith(".parquet"):
            category = file_name.split(".")[0]  
            file_path = os.path.join(MME_path, file_name)
            process_dataset(file_path, category)
    
    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time
    print(f"All categories processed in {total_elapsed_time:.2f} seconds")