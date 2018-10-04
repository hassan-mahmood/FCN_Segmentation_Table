import numpy as np
import torch

def iou(pred, target, n_classes = 2):
  ious = []
  pred = pred.view(-1)
  target = target.view(-1)

  # Ignore IoU for background class ("0")
  for cls in range(1, n_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored
    pred_inds = pred == cls
    target_inds = target == cls
    intersection = (pred_inds[target_inds]).long().sum().data.cpu()[0]  # Cast to long to prevent overflows
    union = pred_inds.long().sum().data.cpu()[0] + target_inds.long().sum().data.cpu()[0] - intersection
    if union == 0:
      ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
    else:
      ious.append(float(intersection) / float(max(union, 1)))
  return np.array(ious)

target=np.array([[1,1,1],[2,2,2],[1,1,1]])
class1=np.array([[1,0,0],[0,1,0],[0,0,0]])
class2=np.array([[0,2,2],[0,0,0],[0,2,0]])
prediction=np.array([class1,class2])
target=torch.from_numpy(target)
prediction=torch.from_numpy(prediction)

# intersection = np.logical_and(target, prediction)
# union = np.logical_or(target, prediction)
# iou_score = np.sum(intersection) / np.sum(union)
# print(iou_score)
print(iou(prediction,target,2))