import numpy as np 

def calculate_iou(preds, labels, num_classes):
    iou_per_class = []
    
    for class_id in range(num_classes):
        intersection = np.logical_and(preds == class_id, labels == class_id).sum()
        union = np.logical_or(preds == class_id, labels == class_id).sum()
        
        if union == 0:
            iou_per_class.append(np.nan)  # Avoid division by zero
        else:
            iou_per_class.append(intersection / union)
    
    return iou_per_class