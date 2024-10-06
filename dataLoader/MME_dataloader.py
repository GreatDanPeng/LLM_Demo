import os
import pandas as pd
from PIL import Image
import io
import torch
from torchvision import transforms

class MMEdataLoader:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.preprocess = transforms.Compose([
            transforms.Resize((4096, 4096)),  
            # transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),          # Convert the image to a PyTorch tensor
            # transforms.Normalize(           
            #     mean=[0.485, 0.456, 0.406],
            #     std=[0.229, 0.224, 0.225]
            # )
        ])

    def load_data(self):
        df = pd.read_parquet(self.dataset_path)
        dataset_dict = df.to_dict(orient='list')  
        dataset_dict['image'] = [self.image_to_tensor(image_dict['bytes']) for image_dict in dataset_dict['image']]
        return dataset_dict

    def bytes_to_image(self, byte_array):
        """ Convert byte array to PIL Image. """
        byte_data = bytes(byte_array)
        image = Image.open(io.BytesIO(byte_data)).convert('RGB')  # Ensure RGB format
        # image.show()
        return image

    def image_to_tensor(self, byte_array):
        """ Convert byte array to a preprocessed PyTorch tensor. """
        image = self.bytes_to_image(byte_array)  # Convert bytes to PIL image
        image_tensor = self.preprocess(image)    # Apply preprocessing (resize, tensor, normalize)
        return image

if __name__ == "__main__":
    dataset_loader = MMEdataLoader('../MME/code_reasoning.parquet')
    mme_dataset = dataset_loader.load_data()
    ## MODIFY BATCH so that each image and question can be feed to chatbot!!!!
    print("finish loading...")
    for i in range(0, 10):
        image = mme_dataset['image'][i]
        question = mme_dataset['question_id'][i]
        category = mme_dataset['category'][i]
        print(f"Loaded {image}")
        print(f"Loaded {question}")
        print(f"Loaded {category}")

    # Check the shape of the first image tensor
    print(f"First image tensor shape: {mme_dataset['image'][0]}")
