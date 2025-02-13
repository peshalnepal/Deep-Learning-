import torch 
from torch import nn
import torchvision.transforms as transform
import cv2
import numpy as np
import argparse
from model import Unet
from PIL import Image

def load_model(model_path, device):
    model = Unet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def preprocess_frame(frame):
    preprocess = transform.Compose([
        transform.ToPILImage(),
        transform.Resize((240, 240)),
        transform.ToTensor(),
        transform.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    return preprocess(frame).unsqueeze(0)

def apply_mask_overlay(image, mask):
    mask_colored = np.zeros_like(image, dtype=np.uint8)
    mask_colored[mask == 255] = [0, 255,0]  # Red color for segmented areas
    overlay = cv2.addWeighted(image, 0.7, mask_colored, 0.3, 0)
    return overlay

def process_image(image_path, model, device):
    image = Image.open(image_path).convert("RGB")
    input_image = preprocess_frame(np.array(image)).to(device)
    
    with torch.no_grad():
        output = model(input_image)
        pred_mask = output.squeeze().cpu().numpy()
    
    pred_mask = (pred_mask > 0.8).astype(np.uint8) * 255
    pred_mask_resized = cv2.resize(pred_mask, (image.width, image.height))
    
    image_np = np.array(image)
    overlay = apply_mask_overlay(image_np, pred_mask_resized)
    
    cv2.imshow("Segmented Overlay", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite("./images/output.jpg",cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_webcam(model, device):
    cap = cv2.VideoCapture(0)  
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break
        
        input_image = preprocess_frame(frame).to(device)
        
        with torch.no_grad():
            output = model(input_image)
            pred_mask = output.squeeze().cpu().numpy()
        
        pred_mask = (pred_mask > 0.8).astype(np.uint8) * 255
        pred_mask_resized = cv2.resize(pred_mask, (frame.shape[1], frame.shape[0]))
        
        overlay = apply_mask_overlay(frame, pred_mask_resized)
        
        cv2.imshow("Segmented Overlay", overlay)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", action="store_true", help="Use image input instead of webcam")
    parser.add_argument("--path", type=str, help="Path to input image", default=None)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model("./saved_model/unet_trained.pth", device)
    
    if args.image and args.path:
        process_image(args.path, model, device)
    else:
        process_webcam(model, device)

if __name__ == "__main__":
    main()
