from Dataset import COCODatasetYOLO
from loss import Loss_
from torch import nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from torchvision.transforms import transforms
from pycocotools.coco import COCO
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter
from model import DarkNet
class ResizeNormalize(object):
    def __init__(self, size=448):
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  
                                 std=[0.229, 0.224, 0.225])
        ])
        
    def __call__(self, image, target):
        return self.transform(image), target

DarkNet_Architecture = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1)
]

def train_yolo(model, train_loader, optimizer, loss_fn, device, epochs):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            predictions = model(images)
            loss = loss_fn(predictions, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}], Loss: {loss.item():.4f}")
        print(f"Epoch [{epoch+1}/{epochs}] Average Loss: {running_loss/len(train_loader):.4f}")
    print("Training completed.")


if __name__ == "__main__":
    epochs = 25
    learning_rate = 1e-4
    batch_size = 40
    split_size = 7
    num_boxes = 2
    num_classes = 11  

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = ResizeNormalize(size=448)

    annotation_file = "./annotations/instances_train2017.json"  # Update this path
    image_dir = "./train2017"                         # Update this path
    dataset = COCODatasetYOLO(annotation_file, image_dir, S=split_size, C=num_classes, transform=transform)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    train_features, train_labels = next(iter(train_loader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    
    img = train_features[0].permute(1, 2, 0).cpu().numpy()
    label = train_labels[0]
    
    plt.imshow(img)
    plt.title("Transformed Image")
    plt.show()
    print(f"Label: {label}")
    
    loss_fn = Loss_(S=split_size, B=num_boxes, C=num_classes)
    model = DarkNet(input_channels=3, Architecture=DarkNet_Architecture,
                    split_size=split_size, num_boxes=num_boxes, num_class=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_yolo(model, train_loader, optimizer, loss_fn, device, epochs)
    
    torch.save(model.state_dict(), "./yolo_model.pth")
