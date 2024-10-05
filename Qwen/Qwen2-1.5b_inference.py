import sys
import os
import json
import time
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

# Import dataloader from data/
sys.path.append(os.path.join(os.path.dirname(__file__), 'data'))
from Huggingface_dataloader import dataLoader

# Load Qwen2-1.5b model from huggingface hub
class QwenInference:
    def __init__(self, model_name="Qwen/Qwen2-1.5B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def generate_answer(self, question):
        inputs = self.tokenizer(question, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer

if __name__ == "__main__":
    total_start_time = time.time()

    # Load GSM8K dataset
    gsm8k_loader = dataLoader('openai/gsm8k', 'main', 'test')
    gsm8k_data = gsm8k_loader.load_data()
    qwen_inference = QwenInference()

    output_path = r'.\output'
    result_file_path = 'GSM8K_result.jsonl'
    output_file = os.path.join(output_path, result_file_path)
    os.makedirs(output_path, exist_ok=True)
    

    with open(result_file_path, 'a') as result_file:
        predictions = []
        references = []
        for QApair in gsm8k_data:
            question = QApair['question']
            reference_answer = QApair['answer']
            start_time = time.time()
            generated_answer = qwen_inference.generate_answer(question)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Generated answer in {elapsed_time:.2f} seconds")

            result = {
                'question': question,
                'generated_answer': generated_answer,
                'reference_answer': reference_answer
            }
            result_file.write(json.dumps(result) + '\n')
            predictions.append(generated_answer)
            references.append(reference_answer)

    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time
    print(f"Total test time: {total_elapsed_time:.2f} seconds") # 26577.31 seconds on a single NVIDIA RTX 3060 Ti 8GB GPU

