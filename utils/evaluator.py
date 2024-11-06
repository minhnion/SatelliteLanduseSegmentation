from sklearn.metrics import precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch
import wandb 

from utils.logging_utils import plot_predictions

def evaluate_on_test_set(model, test_loader, classes, image_dir=None):
    assert model is not None
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)  # Ensure the model is on the correct device

    all_labels = []
    all_preds = []
    all_labels_each_cls = []
    all_preds_each_cls = []

    with torch.no_grad():
        for inputs, masks, _a, _b in test_loader:
            inputs = inputs.to(device)  # Move inputs to the same device as the model
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy().flatten()
            labels = masks.cpu().numpy().flatten()

            plot_predictions(inputs, outputs, masks, epoch=1, num_samples='all', image_dir = image_dir)

            all_preds_each_cls.extend(preds)
            all_labels_each_cls.extend(labels)

            all_preds.extend(preds.flatten())
            all_labels.extend(labels.flatten())

    # Precision and recall mean
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)

    print(f'Test precision: {precision}, Test recall: {recall}')

    log_test = {
        'precision': precision,
        'recall': recall 
    }

    # Precision and recall for each class
    precision_per_class = precision_score(all_labels_each_cls, all_preds_each_cls, average=None, labels=list(range(len(classes))))
    recall_per_class = recall_score(all_labels_each_cls, all_preds_each_cls, average=None, labels=list(range(len(classes))))

    for i, class_name in enumerate(classes):
        if (i==0):
            continue
        print(f'Class: {class_name} - Precision: {precision_per_class[i]:.4f}, Recall: {recall_per_class[i]:.4f}')
        log_test[f'precision_{class_name}'] = precision_per_class[i]
        log_test[f'recall_{class_name}'] = recall_per_class[i]

    # log test
    wandb.log({'Test log': log_test})

    # Confusion Matrix
    plt.figure(figsize=(12, 8))
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(classes))), normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Normalized Confusion Matrix")
    wandb.log({"Confusion-matrix plot": wandb.Image(plt)})
    # plt.show()
