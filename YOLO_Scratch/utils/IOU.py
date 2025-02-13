import torch 
import numpy as np

def intersection_over_union(box1,box2,format="xywh"):
    #box1 (N,4)
    x1=[]
    y1=[]
    x2=[]
    y2=[]
    box1_w=[]
    box1_h=[]
    box2_w=[]
    box2_h=[]
    if format=="xywh":
        box1_x1 = box1[..., 0:1] - 0.5 * box1[..., 2:3]
        box1_y1 = box1[..., 1:2] - 0.5 * box1[..., 3:4]
        box1_x2 = box1[..., 0:1] + 0.5 * box1[..., 2:3]
        box1_y2 = box1[..., 1:2] + 0.5 * box1[..., 3:4]
        
        box2_x1 = box2[..., 0:1] - 0.5 * box2[..., 2:3]
        box2_y1 = box2[..., 1:2] - 0.5 * box2[..., 3:4]
        box2_x2 = box2[..., 0:1] + 0.5 * box2[..., 2:3]
        box2_y2 = box2[..., 1:2] + 0.5 * box2[..., 3:4]
        x1 = torch.max(box1_x1, box2_x1)
        y1 = torch.max(box1_y1, box2_y1)
        x2 = torch.min(box1_x2, box2_x2)
        y2 = torch.min(box1_y2, box2_y2)

        box1_h=(box1_y2 - box1_y1).clamp(min=0)
        box2_w=(box2_x2 - box2_x1).clamp(min=0)
        box1_w=(box1_x2 - box1_x1).clamp(min=0)
        box2_h=(box2_y2 - box2_y1).clamp(min=0)

    else:
        x1=torch.max(box1[...,0:1],box2[...,0:1])
        y1=torch.max(box1[...,1:2],box2[...,1:2])
        x2=torch.min(box1[...,2:3],box2[...,2:3])
        y2=torch.min(box1[...,3:4],box2[...,3:4])
        box1_w=(box1[...,2:3]-box1[...,0:1]).clamp(min=0)
        box1_h=(box1[...,3:4]-box1[...,1:2]).clamp(min=0)
        box2_w=(box2[...,2:3]-box2[...,0:1]).clamp(min=0)
        box2_h=(box2[...,3:4]-box2[...,1:2]).clamp(min=0)
    
    Area_of_intersection=(x2-x1).clamp(min=0)*(y2-y1).clamp(min=0)
    Area_of_union=box1_w*box1_h+box2_w*box2_h-Area_of_intersection+1e-6
    return Area_of_intersection/Area_of_union