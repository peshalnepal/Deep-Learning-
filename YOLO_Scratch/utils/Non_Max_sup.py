import torch
from .IOU import intersection_over_union
def Non_max(prediction,iou_threshold,prob_threshold,format="xyxy"):
    #prediction of bounding box [[class,probability, x1,y1,x2,y2],[class,probability, x1,y1,x2,y2],[class,probability, x1,y1,x2,y2]]
    assert(type(prediction))==list
    bboxes=[box for box in prediction if box[1]>prob_threshold]
    bboxes=sorted(bboxes,key=lambda x:x[1],reverse=True)
    
    bboxes_nms=[]
    while bboxes:
        chosen_box=bboxes.pop(0)
        bboxes=[
            box 
            for box in bboxes
            if box[0]!=chosen_box[0]
            or intersection_over_union(torch.Tensor(chosen_box[2:]),torch.Tensor(box[2:]),format=format)<iou_threshold
        ]
        bboxes_nms.append(chosen_box)

    return bboxes_nms