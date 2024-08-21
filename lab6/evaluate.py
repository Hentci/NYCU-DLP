import os
import json
import torch
import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image
from PIL import Image

# Assuming evaluation_model is defined as in the provided code
from evaluator import evaluation_model

import numpy as np
import random


seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.random.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

# Function to denormalize the images for visualization
def denormalize(tensor):
    return tensor * torch.tensor((0.5, 0.5, 0.5)).view(3, 1, 1) + torch.tensor((0.5, 0.5, 0.5)).view(3, 1, 1)

# Load test.json and define object_mapping
with open('./new_test.json', 'r') as f:
    test_data = json.load(f)

object_mapping = {"gray cube": 0, "red cube": 1, "blue cube": 2, "green cube": 3, 
                  "brown cube": 4, "purple cube": 5, "cyan cube": 6, "yellow cube": 7, 
                  "gray sphere": 8, "red sphere": 9, "blue sphere": 10, "green sphere": 11, 
                  "brown sphere": 12, "purple sphere": 13, "cyan sphere": 14, "yellow sphere": 15, 
                  "gray cylinder": 16, "red cylinder": 17, "blue cylinder": 18, "green cylinder": 19, 
                  "brown cylinder": 20, "purple cylinder": 21, "cyan cylinder": 22, "yellow cylinder": 23}

# Function to convert labels to one-hot encoding
def labels_to_one_hot(labels, object_mapping):
    one_hot = torch.zeros(len(object_mapping))
    for label in labels:
        one_hot[object_mapping[label]] = 1
    return one_hot

# Directory containing the generated images
image_dir = '/home/hentci/code/NYCU-DLP/lab6/generated_images'

# Load and preprocess images
images = []
labels = []

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Iterate over the test data and corresponding image files
for idx, label_list in enumerate(test_data):
    print(idx, ':',label_list)
    img_name = f"sample_{idx}.png"  
    img_path = os.path.join(image_dir, img_name)
    
    if os.path.exists(img_path):
        img = Image.open(img_path).convert("RGB")
        img = transform(img)
        images.append(img)
    
        one_hot_label = labels_to_one_hot(label_list, object_mapping)
        labels.append(one_hot_label)
    else:
        print(f"Image {img_name} not found in {image_dir}.")

# # Convert lists to tensors
# images = torch.stack(images)
# # # images = images.clip(-1, 1)
# labels = torch.stack(labels)

print(images)

# Instantiate the evaluator model
evaluator = evaluation_model()

# Evaluate the accuracy
accuracy = evaluator.eval(images.cuda(), labels.cuda())
print(f"Accuracy of the synthetic images: {accuracy * 100:.2f}%")

# Denormalize and save images as a grid
denorm_images = denormalize(images)
grid = make_grid(denorm_images, nrow=8)
save_image(grid, 'synthesized_images_grid.png')
print('Synthesized images saved as "synthesized_images_grid.png"')