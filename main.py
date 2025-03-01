from torch.utils.data import DataLoader
from torchvision import transforms

import numpy as np
import pandas as pd

from config import CONFIG
from train import train
from test import test
from model import ResNet18
from dataset import MNISTLaplacianDataset, MedMNISTDataset

from collections import defaultdict

import warnings
warnings.filterwarnings("ignore")

################
# Load Dataset #
################
data = np.load(CONFIG['data_path'])
train_images, train_labels = data['train_images'], data['train_labels']
test_images, test_labels = data['test_images'], data['test_labels']

########
# Main #
########
# Define transformation
data_transform = transforms.Compose([transforms.ToTensor()])

# Initialize dictionaries to store results
results = defaultdict(lambda: defaultdict(list))
classwise_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

epsilon_values = [0.01, 0.1, 1.0, 10, 100, 1000]
num_folds = CONFIG['num_folds']

for epsilon_p in epsilon_values:
    print(f"Running for epsilon_p: {epsilon_p}")
    
    for fold in range(num_folds):
        print(f"Fold {fold + 1}/{num_folds}")
        
        # Create datasets
        train_dataset = MNISTLaplacianDataset(train_images, train_labels, epsilon_p=epsilon_p, transform=data_transform)
        test_dataset = MedMNISTDataset(test_images, test_labels, transform=data_transform)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

        # Initialize model
        model = ResNet18(CONFIG['num_channels'], CONFIG['num_classes']).to(CONFIG['device'])

        # Train and test the model
        train_model = train(model, train_loader, CONFIG['task'])
        acc, auc, class_acc = test(train_model, test_loader, CONFIG['task'])
        
        # Store results
        results[epsilon_p]['acc'].append(acc)
        results[epsilon_p]['auc'].append(auc)
        for k, v in class_acc.items():
            classwise_results[epsilon_p][k]['acc'].append(v)

        del model, train_model, train_dataset, test_dataset, train_loader, test_loader

# Convert results to pandas DataFrame
overall_results = []
for epsilon_p in epsilon_values:
    overall_results.append({
        "Epsilon": epsilon_p,
        "Accuracy Mean": np.mean(results[epsilon_p]['acc']),
        "Accuracy Std": np.std(results[epsilon_p]['acc']),
        "AUC Mean": np.mean(results[epsilon_p]['auc']),
        "AUC Std": np.std(results[epsilon_p]['auc'])
    })

df_overall = pd.DataFrame(overall_results)
print("\nOverall Results:")
print(df_overall.to_string(index=False))

# Convert class-wise results to pandas DataFrame
classwise_results_list = []
for epsilon_p in epsilon_values:
    for k in classwise_results[epsilon_p]:
        classwise_results_list.append({
            "Epsilon": epsilon_p,
            "Class": k,
            "Accuracy Mean": np.mean(classwise_results[epsilon_p][k]['acc']),
            "Accuracy Std": np.std(classwise_results[epsilon_p][k]['acc'])
        })

df_classwise = pd.DataFrame(classwise_results_list)
print("\nClass-wise Results:")
print(df_classwise.to_string(index=False))

# Save results to CSV
df_overall.to_csv(f"{CONFIG['result_path']}overall_results.csv", index=False)
df_classwise.to_csv(f"{CONFIG['result_path']}classwise_results.csv", index=False)