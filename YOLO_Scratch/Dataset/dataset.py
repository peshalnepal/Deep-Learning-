import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from pycocotools.coco import COCO
from torchvision.transforms import ToTensor
from collections import defaultdict

class COCODatasetYOLO(Dataset):
    def __init__(self, 
                 annotation_file, 
                 image_dir, 
                 S=7, 
                 C=11, 
                 transform=None):
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())
        self.image_dir = image_dir
        self.S = S
        self.C = C
        self.transform = transform

        cats = self.coco.loadCats(self.coco.getCatIds())
        cats = sorted(cats, key=lambda x: x['id'])
        self.supercategory_map= defaultdict(int)
        self.cat2label = {cat['id']:cat["supercategory"]  for i, cat in enumerate(cats)}
        idx=0
        for cat in cats:
            supercategory = cat["supercategory"]
            if supercategory not in self.supercategory_map:  # Check keys, not items
                self.supercategory_map[supercategory] = idx
                idx += 1  # Increment only when a new category is added
        
    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.imgs[image_id]
        file_name = image_info['file_name']
        image_path = os.path.join(self.image_dir, file_name)
        image = Image.open(image_path).convert("RGB")
        
        img_width = image_info['width']
        img_height = image_info['height']
        
        target = torch.zeros((self.S, self.S, self.C + 5))
        
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        
        for ann in anns:
            if ann.get('iscrowd', 0) == 1:
                continue
            bbox = ann['bbox']
            if bbox[2] <= 0 or bbox[3] <= 0:
                continue  # skip invalid boxes

            x, y, w, h = bbox

            # Calculate the center of the bounding box in pixel coordinates.
            cx = x + w / 2
            cy = y + h / 2

            # Normalize center coordinates to [0, 1]
            cx_norm = cx / img_width
            cy_norm = cy / img_height
            # Normalize width and height relative to the image dimensions.
            w_norm = w / img_width
            h_norm = h / img_height

            # Determine in which grid cell the center falls.
            # (Make sure to clip indices if center == 1 exactly)
            grid_i = int(cx_norm * self.S)
            grid_j = int(cy_norm * self.S)
            if grid_i >= self.S:
                grid_i = self.S - 1
            if grid_j >= self.S:
                grid_j = self.S - 1

            # Calculate offsets relative to the top-left corner of the grid cell.
            # For example, if cx_norm * S = 3.4, then offset is 0.4.
            x_cell = cx_norm * self.S - grid_i
            y_cell = cy_norm * self.S - grid_j

            # Determine class index.
            category_id = ann['category_id']
            super_Category = self.cat2label[category_id]
            class_idx=self.supercategory_map[super_Category]
            if target[grid_i, grid_j, self.C] == 0:
                # Set the one-hot class vector.
                target[grid_i, grid_j, class_idx] = 1.0
                # Set objectness (confidence) to 1.
                target[grid_i, grid_j, self.C] = 1.0
                # Set bounding box coordinates: (x_cell, y_cell, w_norm, h_norm)
                target[grid_i, grid_j, self.C + 1] = x_cell
                target[grid_i, grid_j, self.C + 2] = y_cell
                target[grid_i, grid_j, self.C + 3] = w_norm
                target[grid_i, grid_j, self.C + 4] = h_norm
            # If more than one object falls in the same cell, you might want to
            # choose the one with a higher area or ignore the subsequent ones.
            # This implementation simply uses the first encountered annotation.

        if self.transform:
            image, target = self.transform(image, target)
        else:
            image = ToTensor()(image)

        return image, target


# annotation_file = "/home/peshal/programming/Deep Learning/annotations/instances_train2017.json"  # Update this path
# image_dir = "/home/peshal/programming/Deep Learning/train2017"   
# dataset = COCODatasetYOLO(annotation_file, image_dir, S=7, C=11)
# image, target = dataset[0]
# print(image.shape)      # e.g., (3, H, W)
# print(target.shape)     # (7, 7, 85)  because 80 + 5 = 85
