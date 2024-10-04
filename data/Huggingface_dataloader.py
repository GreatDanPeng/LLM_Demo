import os
from datasets import load_dataset

class dataLoader:
    def __init__(self, dataset_name, config_name=None, split=None):
        self.dataset_name = dataset_name
        self.config_name = config_name
        self.split = split

    def load_data(self):
        if self.config_name and self.split:
            dataset = load_dataset(self.dataset_name, self.config_name, split=self.split)
        elif self.config_name:
            dataset = load_dataset(self.dataset_name, self.config_name)
        else:
            dataset = load_dataset(self.dataset_name)
        return dataset

# Template 
# if __name__ == "__main__":
#     # Load GSM8K dataset
#     gsm8k_loader = dataLoader('openai/gsm8k', 'main', 'test')
#     gsm8k_data = gsm8k_loader.load_data()
#     print(f"Loaded {len(gsm8k_data)} GSM8K records")