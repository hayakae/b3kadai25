import os
import clip
import torch
from torchvision.datasets import Food101
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torch.utils.data import DataLoader
from tqdm import tqdm

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# Download and load the Food101 dataset (test data)
dataset = Food101(root=os.path.expanduser("./data"), split="test", download=True)

# Preprocessing for the dataset
preprocess_transform = Compose([
    Resize((224, 224)),
    CenterCrop(224),
    ToTensor(),
    Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
])

# DataLoader with batch processing
dataloader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=False,
    num_workers=4,
    collate_fn=lambda x: (torch.stack([preprocess_transform(img) for img, _ in x]),
                          torch.tensor([label for _, label in x]))
)

# Create text prompts for each class
text_inputs = torch.cat([clip.tokenize(f"a photo of {c}") for c in dataset.classes]).to(device)

# Precompute text features
with torch.no_grad():
    text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# Prepare for evaluation
correct_top1 = 0
correct_top5 = 0
total_samples = len(dataset)

# Evaluate on the dataset
with torch.no_grad():
    for images, labels in tqdm(dataloader, desc="Evaluating"):
        # Move data to device
        images = images.to(device)
        labels = labels.to(device)

        # Compute image features
        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # Compute similarity and find top-k predictions
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity.topk(5, dim=-1)  # Top-5 predictions

        # Check Top-1 and Top-5 accuracy
        for label, top5 in zip(labels, indices):
            if label in top5:
                correct_top5 += 1
                if label == top5[0]:
                    correct_top1 += 1

# Calculate and print results
top1_accuracy = correct_top1 / total_samples * 100
top5_accuracy = correct_top5 / total_samples * 100

print(f"Top-1 Accuracy: {top1_accuracy:.2f}%")
print(f"Top-5 Accuracy: {top5_accuracy:.2f}%")
