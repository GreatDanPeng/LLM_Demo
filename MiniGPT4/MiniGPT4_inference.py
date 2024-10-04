import argparse
import os
import sys
import random
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from transformers import StoppingCriteriaList

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'MiniGPT-4'))
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2, StoppingCriteriaSub
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data'))
from Huggingface_dataloader import dataLoader

# Reduce GPU memory
import gc
gc.collect()
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.9, max_split_size_mb:128"

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


def main():
    args = parse_args()
    cfg = Config(args)
    setup_seeds(cfg)

    conv_dict = {'pretrain_vicuna0': CONV_VISION_Vicuna0,
                 'pretrain_llama2': CONV_VISION_LLama2}

    print('Initializing Chat')
    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

    CONV_VISION = conv_dict[model_config.model_type]

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    stop_words_ids = [[835], [2277, 29937]]
    stop_words_ids = [torch.tensor(ids).to(device='cuda:{}'.format(args.gpu_id)) for ids in stop_words_ids]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id), stopping_criteria=stopping_criteria)
    print('Initialization Finished')

    ## Apply MiniGPT to the MME dataset
    ### MME data preprocess
    mme_loader = dataLoader("lmms-lab/MME")
    mme_dataset = mme_loader.load_data()
    results_by_category = {}
    total_start_time = time.time()

    for item in mme_dataset['test']:
        image_id = item['image_id']
        question = item['question']
        category = item['category']

        if category not in results_by_category:
            results_by_category[category] = {
                'results': [],
                'start_time': time.time()
            }

        answer = chat.answer(conv=CONV_VISION,
                             img_list=[image_id],
                             num_beams=1,
                             temperature=1.0,
                             max_new_tokens=300,
                             max_length=2000)[0]

        results_by_category[category]['results'].append({
            'image_id': image_id,
            'question': question,
            'answer': answer
        })

    for category, data in results_by_category.items():
        file_path = f"{category}.txt"
        with open(file_path, 'w') as file:
            for result in data['results']:
                file.write(f"{result['image_id']}\t{result['question']}\t{result['answer']}\n")
        end_time = time.time()
        elapsed_time = end_time - data['start_time']
        print(f"Category '{category}' processed in {elapsed_time:.2f} seconds")

    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time
    print(f"Total processing time: {total_elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()