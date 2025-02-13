from loss import Loss_
import torch
from PIL import Image
import numpy as np
from collections import Counter
from model import DarkNet
import cv2
from utils import convert_cellboxes,plot_image
from torchvision import transforms


class ResizeNormalize:
    def __init__(self, size=448):
        self.size = size
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # ImageNet Mean
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)   # ImageNet Std

    def __call__(self, image):
        """
        :param image: Input image in OpenCV format (numpy array)
        :return: Normalized tensor
        """
        image = cv2.resize(image, (self.size, self.size))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        image = torch.from_numpy(image).permute(2, 0, 1)  

        return image

model_Architecture = [
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

def evaluate_webcam(model, transform, device):
    capture=cv2.VideoCapture(0)
    model.to(device)
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break  
        _,frame=capture.read()
        image=transform(frame)
        image = image.to(device)
        image=image.unsqueeze(0)
        predictions = model(image)
        boxes=convert_cellboxes(predictions=predictions,S=7,C=11,B=2,device=device)
        frame=plot_image(cv2.resize(frame, (448, 448)),boxes=boxes[0],threshold=0.7)
        cv2.imshow('Video Capture', frame) 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()


def evaluate_image(image_path, model, device,transform):
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError("Image not found or unable to read")

    
    image = transform(frame)
    image = image.to(device)
    image = image.unsqueeze(0)  
    model.to(device)
    predictions = model(image)
    boxes = convert_cellboxes(predictions=predictions, S=7, C=11, B=2, device=device)
    frame = plot_image(cv2.resize(frame, (448, 448)), boxes=boxes[0], threshold=0.5)
    cv2.imshow('Image Detection', frame)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

if __name__ == "__main__":

    split_size = 7
    num_boxes = 2
    num_classes = 11  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = ResizeNormalize(size=448)
    
    
    loss_fn = Loss_(S=split_size, B=num_boxes, C=num_classes)
    model = DarkNet(input_channels=3, Architecture=model_Architecture,
                    split_size=split_size, num_boxes=num_boxes, num_class=num_classes)
    checkpoint_path="./saved_model/checkpoint_epoch_5.pth"
    checkpoint = torch.load(checkpoint_path,map_location=torch.device('cuda:0'),weights_only=True)  # Change to 'cuda' if using GPU
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    model.eval()

    model.eval()
    # Train the model.
    evaluate_webcam(model, transform=transform, device=device)
    # evaluate_image("/home/peshal/programming/YOLO_Scratch/test_image.jpg",model,transform=transform,device=device)
    
