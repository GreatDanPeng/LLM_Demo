# import torch
# import os
# cuda_id = torch.cuda.current_device()
# print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
# print(f"CUDA version: {torch.version.cuda}")
# print(f"ID of current CUDA device:{torch.cuda.current_device()}")
# print(f"Name of current CUDA device:{torch.cuda.get_device_name(cuda_id)}")
# print(os.path.dirname(__file__))

import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Set environment variable to manage GPU memory
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.9,max_split_size_mb:128"

# Load the MME dataset (first category only)
def load_mme_first_category():
    # Assuming the dataset is in a JSON format or similar
    # Replace this with actual loading code
    dataset = [
        {"image_id": "41036.jpg", "question": "Is this artwork displayed in kunsthistorisches museum, vienna?", "category": "museum"},
        {"image_id": "41036.jpg", "question": "Is this artwork displayed in national museum of art, minsk?", "category": "museum"},
        # Add more entries as needed
    ]
    return dataset

# Initialize the model and tokenizer
def initialize_model():
    model_name = "path/to/pretrained_minigpt4"  # Replace with actual model path
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

# Run inference on the first category
def run_inference(tokenizer, model, dataset):
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    with torch.no_grad():
        for item in dataset:
            inputs = tokenizer(item["question"], return_tensors="pt").to(device)
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=-1).item()
            answer = "Yes" if prediction == 1 else "No"
            results.append({
                "image_id": item["image_id"],
                "question": item["question"],
                "answer": answer
            })
    return results

# Save results to a file
def save_results(results, output_file):
    with open(output_file, 'w') as file:
        for result in results:
            file.write(f"{result['image_id']}\t{result['question']}\t{result['answer']}\n")

def main():
    dataset = load_mme_first_category()
    tokenizer, model = initialize_model()
    results = run_inference(tokenizer, model, dataset)
    save_results(results, "mme_first_category_results.txt")

if __name__ == "__main__":
    main()