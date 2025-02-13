import torch
from collections import Counter
from .IOU import intersection_over_union

def meam_avg_precision(prediction,target,iou_threshold=0.5,num_classes=11,format="xywh"):
    #prediction format for a batch is [[training_idx,class,prob,x1,y1,x2,y2],......]
    average_mean_Precision=[]
    epsilon=1e-6
    for class_no in num_classes:
        predicted_class=[]
        ground_truth=[]
        for predict in prediction:
            if predict[1]==class_no:
                predicted_class.append(predict)
        for true_value in target:
            if true_value[1]==class_no:
                ground_truth.append(true_value)
        predicted_class.sort(lambda x:x[2],reverse=True)
        num_of_boxes=Counter(value[0] for value in ground_truth)
        for key, val in num_of_boxes.items():
            num_of_boxes[key]=torch.zeros(val)
        TP=torch.zeros(len(prediction))
        FP=torch.zeros(len(prediction))
        for idx,detected in enumerate(predicted_class):
            num_gt_for_detected=0
            best_iou=0
            best_gt=-1
            for gt in ground_truth:
                if gt[0]==detected[0]:
                    iou=intersection_over_union(gt[3:],detected[3:],format=format)
                    if iou>best_iou:
                        best_iou=iou
                        best_gt=num_gt_for_detected
                    num_gt_for_detected+=1
            if best_iou>iou_threshold:
                if num_of_boxes[detected[0]][best_gt]==0:
                    TP[idx]=1
                    num_of_boxes[detected[0]][best_gt]=1
                else:
                    FP[idx]=1
            else:
                FP[idx]=1
            
        TP_cumulative = torch.cumsum(TP, dim=0)
        FP_cumulative = torch.cumsum(FP, dim=0)
        recalls = TP_cumulative / (len(ground_truth) + epsilon)
        precisions = TP_cumulative / (TP_cumulative + FP_cumulative + epsilon)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        average_mean_Precision.append(torch.trapz(precisions, recalls))

    return sum(average_mean_Precision) / len(average_mean_Precision)