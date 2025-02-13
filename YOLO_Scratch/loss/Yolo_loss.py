from utils import intersection_over_union
import torch
from torch import nn

class Loss_(nn.Module):
    def __init__(self, S=7, B=2, C=11):
        super(Loss_, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.cross_entropy = nn.CrossEntropyLoss()
        self.S = S
        self.C = C
        self.B = B
        self.lambdanoObj = 0.5
        self.lambdacord = 5

    def forward(self, prediction, target):
        # Reshape prediction to (batch, S, S, C+5*B)
        prediction = prediction.reshape(-1, self.S, self.S, self.C + self.B * 5)
        # Note: target is assumed to already have shape (batch, S, S, C+5)

        # Calculate IoU for each of the two predicted boxes versus the target box.
        iou_box1 = intersection_over_union(
            prediction[..., self.C + 1:self.C + 5],
            target[..., self.C + 1:self.C + 5]
        )
        iou_box2 = intersection_over_union(
            prediction[..., self.C + 6:self.C + 10],
            target[..., self.C + 1:self.C + 5]
        )
        # Concatenate IoUs along a new dimension and get the best IoU and corresponding box index.
        iou_cat = torch.cat([iou_box1.unsqueeze(0), iou_box2.unsqueeze(0)], dim=0)
        iou_max, best_box = torch.max(iou_cat, dim=0)
        best_box = best_box.unsqueeze(-1)  
        # exist_box indicates whether an object exists in a cell (shape: batch x S x S x 1)
        exist_box = target[..., self.C].unsqueeze(3)

        # Select the predicted box based on best_box:
        # If best_box==1, use prediction[..., C+6:C+10]; else use prediction[..., C+1:C+5]
        box_prediction = exist_box * (
            (best_box * prediction[..., self.C + 6:self.C + 10]) +
            ((1 - best_box) * prediction[..., self.C + 1:self.C + 5])
        )
        box_targets = exist_box * target[..., self.C + 1:self.C + 5]

        # Instead of doing in-place sqrt modifications, compute new tensors:
        # For width and height, take the square root (with a small epsilon for numerical stability).
        pred_wh = torch.sign(box_prediction[..., 2:4]) * (torch.sqrt((box_prediction[..., 2:4]+1e-6)) )
        target_wh = torch.sqrt(box_targets[..., 2:4]+1e-6)
        # Reconstruct the box predictions and targets by concatenating the (x, y) with the transformed (w, h)
        box_prediction_reg = torch.cat((box_prediction[..., :2], pred_wh), dim=-1)
        box_targets_reg = torch.cat((box_targets[..., :2], target_wh), dim=-1)

        # Compute the bounding box loss using the mean-squared error.
        box_loss = self.mse(
            torch.flatten(box_prediction_reg, end_dim=-2),
            torch.flatten(box_targets_reg, end_dim=-2)
        )

        # For the confidence score, select the appropriate prediction based on best_box.
        pred_box = best_box * prediction[..., self.C + 5:self.C + 6] + (1 - best_box) * prediction[..., self.C:self.C + 1]
        targ_box=best_box * target[..., self.C:self.C + 1] + (1 - best_box) * target[..., self.C:self.C + 1]
        object_loss = self.mse(
            torch.flatten(exist_box * pred_box),
            torch.flatten(exist_box * targ_box)
        )

        # Loss for cells without objects.
        no_object_loss = self.mse(
            torch.flatten((1 - exist_box) * prediction[..., self.C + 5:self.C + 6], start_dim=1),
            torch.flatten((1 - exist_box) * target[..., self.C:self.C + 1], start_dim=1)
        )
        no_object_loss += self.mse(
            torch.flatten((1 - exist_box) * prediction[..., self.C:self.C + 1], start_dim=1),
            torch.flatten((1 - exist_box) * target[..., self.C:self.C + 1], start_dim=1)
        )
        not_responsible_conf_1 = best_box * prediction[..., self.C : self.C+1]      # box0 is not chosen if best_box=1
        not_responsible_conf_2 = (1 - best_box) * prediction[..., self.C+5 : self.C+6]

        no_object_loss_obj_cells = self.mse(
            torch.flatten(exist_box * not_responsible_conf_1),
            torch.flatten(exist_box * torch.zeros_like(not_responsible_conf_1))
        )
        no_object_loss_obj_cells += self.mse(
            torch.flatten(exist_box * not_responsible_conf_2),
            torch.flatten(exist_box * torch.zeros_like(not_responsible_conf_2))
        )
        no_object_loss+=no_object_loss_obj_cells
        # For class loss, use CrossEntropyLoss.
        # Permute the predictions to shape (batch, C, S, S) and take the argmax of the target classes.
        class_loss = self.cross_entropy(
            (exist_box * prediction[..., :self.C]).permute(0, 3, 1, 2),
            torch.argmax(exist_box * target[..., :self.C], dim=-1)
        )

        loss = (self.lambdacord * box_loss + object_loss +
                self.lambdanoObj * no_object_loss + class_loss)
        return loss
