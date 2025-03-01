import torch

import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.special import softmax

from config import CONFIG

device = CONFIG['device']

def test(model, test_loader, task):
    model.eval()
    correct = 0
    total = 0
    all_targets = []
    all_outputs = []
    class_correct = {}
    class_total = {}

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.float().to(device), targets.to(device)

            outputs = model(inputs)

            if task == 'multi-label, binary-class':
                targets = targets.to(float().torch.float32)
                predicted = (outputs > 0.5).float()
                correct += (predicted == targets).all(dim=1).sum().item()
            else:
                targets = targets.squeeze().long()
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                
                # Calculate class-wise accuracy
                for true, pred in zip(targets, predicted):
                    if true.ndim == 0:
                        true = int(true.item())
                        pred = int(pred.item())
                    else:
                        true = tuple(true.cpu().numpy())
                        pred = tuple(pred.cpu().numpy())
                    
                    if true == pred:
                        class_correct[true] = class_correct.get(true, 0) + 1
                    class_total[true] = class_total.get(true, 0) + 1

            total += targets.size(0)
            all_targets.extend(targets.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())

    acc = 100. * correct / total
    class_acc = {k: 100. * v / class_total[k] for k, v in class_correct.items()}

    auc = compute_auc(all_targets, all_outputs, task)

    print(f'Accuracy: {acc:.2f}%')  
    print(f'AUC: {auc:.4f}')
    print('Class-wise accuracy:')
    for k, v in class_acc.items():
        print(f'Class {k}: {v:.2f}%')

    return acc, auc, class_acc

####################
# Metric Functions #
####################
def compute_auc(all_targets, all_outputs, task):
    all_targets, all_outputs = np.array(all_targets), np.array(all_outputs)
    if task == 'multi-label, binary-class':
        return roc_auc_score(all_targets, all_outputs, average='macro')
    elif all_outputs.shape[1] == 2:
        return roc_auc_score(all_targets, all_outputs[:, 1])
    else:
        softmax_outputs = softmax(all_outputs, axis=1)
        return roc_auc_score(all_targets, softmax_outputs, multi_class='ovr', average='macro')