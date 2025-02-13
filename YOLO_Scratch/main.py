import torch
from PIL import Image
import numpy as np
from collections import Counter
from model import DarkNet
import cv2
import argparse
from torchvision import transforms

from utils import convert_cellboxes,plot_image

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

def test_webcam(model, transform, device):
    capture=cv2.VideoCapture(0)
    model.to(device)
    with torch.no_grad():
        while capture.isOpened():
            ret, frame = capture.read()
            if not ret:
                break 
            _,frame=capture.read()
            image=transform(frame)
            image = image.to(device)
            image=image.unsqueeze(0)
            print(image.shape) 

            predictions = model(image)
            boxes=convert_cellboxes(predictions=predictions,S=7,C=11,B=2,device=device)
            frame=plot_image(frame,boxes=boxes[0],threshold=0.7)
            cv2.imshow('Video Capture', frame) 
            cv2.imwrite("./images/webcam_output.jpg", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        capture.release()
        cv2.destroyAllWindows()

def test_image(image_path, model,transform,device):
    model.to(device)
    with torch.no_grad():
        frame = cv2.imread(image_path)
        frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame is None:
            raise ValueError("Image not found or unable to read")
        
        image = transform(frame)
        image = image.to(device)
        image = image.unsqueeze(0) 
        

        predictions = model(image)
        boxes = convert_cellboxes(predictions=predictions, S=7, C=11, B=2, device=device)
        frame = plot_image(frame, boxes=boxes[0], threshold=0.7)
        frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imwrite("./images/output.jpg", frame)
        cv2.imshow('Image Detection', frame)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO Object Detection")
    parser.add_argument("--webcam", action="store_true", help="Use webcam for real-time object detection")
    parser.add_argument("--image", action="store_true", help="Use an image for object detection")
    parser.add_argument("--img_pth", type=str, help="Path to the image file (default: images/test_image.jpg)")

    args = parser.parse_args()

    # Default values
    DEFAULT_IMAGE_PATH = "images/test_image.jpg"

    split_size = 7
    num_boxes = 2
    num_classes = 11  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = ResizeNormalize(size=448)

    model = DarkNet(input_channels=3, Architecture=model_Architecture,
                    split_size=split_size, num_boxes=num_boxes, num_class=num_classes)
    checkpoint_path = "./saved_model/yolo_model.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    

    model.eval()

    if args.image:
        img_path = args.img_pth if args.img_pth else DEFAULT_IMAGE_PATH
        test_image("/home/peshal/programming/YOLO_Scratch/images/test_image.jpg", model, transform, device)
    elif args.webcam:
        test_webcam(model, transform, device)
    else:
        print("[INFO] No arguments provided. Defaulting to webcam mode.")
        test_webcam(model, transform, device)
