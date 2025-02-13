
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from .Non_Max_sup import Non_max

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import cv2

ccoco_supercategories = ["person","vehicle","outdoor","animal","accessory","sports","kitchen","food","furniture","electronic","appliance","indoor"]
def convert_cellboxes(predictions, S=7,C=20,B=2,device="cpu"):

    predictions = predictions.to(device)
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, S, S, C+B*5)

    all_boxes = []
    for b in range(batch_size):
        boxes = []
        for i in range(S):      
            for j in range(S):
                cell_pred = predictions[b, i, j]
                class_scores = cell_pred[:C]
                predicted_class = int(torch.argmax(class_scores).item())
                print(torch.max(class_scores))
                conf1 = cell_pred[C].item()
                box1 = cell_pred[C+1:C+5]
                conf2 = cell_pred[C+5].item()
                box2 = cell_pred[C+6:C+10]

                if conf1 > conf2:
                    best_conf = conf1
                    best_box = box1
                else:
                    best_conf = conf2
                    best_box = box2

                # Convert cell-relative bounding box to image-relative.
                # best_box: [x_cell, y_cell, w_cell, h_cell] (each in [0,1])
                x_cell, y_cell, w_cell, h_cell = best_box.tolist()
                x_img = (j + x_cell) / S  # j is the cell column index
                y_img = (i + y_cell) / S  # i is the cell row index
                w_img = w_cell
                h_img = h_cell 

                boxes.append([predicted_class,torch.max(class_scores), best_conf, x_img, y_img, w_img, h_img])
        all_boxes.append(boxes)

    return all_boxes


def plot_image(image, boxes, threshold=0.5):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy() 
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    if image.shape[0] == 3: 
        image = np.transpose(image, (1, 2, 0)) 
    height, width, _ = image.shape
    color = (255, 0, 0)
    thickness = 2

    for box in boxes:
        prediction,predicted_prob, best_conf, x_center, y_center, box_width, box_height = box
        if isinstance(box, torch.Tensor):
            box = box.detach().cpu().numpy()
        x1 = int((x_center - box_width / 2) * width)
        y1 = int((y_center - box_height / 2) * height)
        w_pixels = int(box_width * width)
        h_pixels = int(box_height * height)
        if best_conf > threshold:
            image = cv2.rectangle(image, (x1, y1), (x1 + w_pixels, y1 + h_pixels), color, thickness)
            image = cv2.rectangle(image, (x1, y1-20), (x1 + w_pixels, y1), color, -1)
            image = cv2.putText(img=image, text=ccoco_supercategories[prediction],  org=(x1, y1 - 5),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.5, color=(255, 255, 255),thickness=1, lineType=cv2.LINE_AA)
    return image


def get_bboxes(loader, model, iou_threshold, threshold, pred_format="cells",
               box_format="midpoint", device="cuda"):
    all_pred_boxes = []
    all_true_boxes = []
    model.eval() 
    image_index = 0  

    for x, labels in loader:
        x = x.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        true_boxes_batch = convert_cellboxes(labels)      
        pred_boxes_batch = convert_cellboxes(predictions) 

        for i in range(batch_size):
            nms_boxes = Non_max(
                pred_boxes_batch[i],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )
            for box in nms_boxes:
                all_pred_boxes.append([image_index] + box)
            
            for box in true_boxes_batch[i]:
                if box[1] > threshold:
                    all_true_boxes.append([image_index] + box)
            
            image_index += 1

    model.train()  
    return all_pred_boxes, all_true_boxes

if __name__ == "__main__":
    batch_size = 1
    S = 7
    dummy_predictions = torch.rand(batch_size, S * S * 30)

    boxes = convert_cellboxes(dummy_predictions, S=S)

    print("Formatted bounding boxes:")
    for img_idx, img_boxes in enumerate(boxes):
        print(f"Image {img_idx}:")
        for box in img_boxes:
            print(box)


