import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import CONFIG
from model import ResNet18
from dataset import MedMNISTDataset

import tqdm

from collections import defaultdict

import importlib.util

spec = importlib.util.spec_from_file_location("test", "./test.py")
test_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test_module)
test = test_module.test  # Use test function/class from test.py

import warnings
warnings.filterwarnings("ignore")

###########
# Dataset #
###########
#? LAPLACIAN NOISE BASED DATASET
class MNISTLaplacianDataset(Dataset):
    def __init__(self, images, labels, epsilon_p, transform):
        """
        Args:
            images: Original image dataset
            labels: Corresponding labels
            epsilon_p: Standard deviation of the Gaussian noise to be added
            transform: Optional transform to be applied on the noisy image
        """
        self.images = images
        self.labels = labels
        self.epsilon_p = epsilon_p
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Apply Gaussian noise
        noisy_image = self.add_l2_laplace_noise(image, self.epsilon_p)

        if self.transform:
            noisy_image = self.transform(noisy_image)
                
        return noisy_image, label
    
    def sample_l2lap(self, eta:float, d:int) -> np.array:
        """
            Returns
            d dimensional noise sampled from `L2 laplace'
            https://math.stackexchange.com/questions/3801271/sampling-from-a-exponentiated-multivariate-distribution-with-l2-norm
        """
        R = np.random.gamma(d, scale = 1.0/eta)
        Z = np.random.normal(0, 1, size = d)
        return R * (Z / np.linalg.norm(Z)) #shape is (d,) one dimensional array
    
    def add_l2_laplace_noise(self, image, epsilon_p):
        """Adds L2 Laplace noise to the image based on the given epsilon_p."""
        shape = image.shape
        d = np.prod(shape)
        k = 0.01 # <= e^epsilon_p
        eta = (epsilon_p / 2) - np.log(np.sqrt(k))
        noise = self.sample_l2lap(eta, d).reshape(shape)
        noisy_image = np.clip(image + noise, 0, 255)
        noisy_image = noisy_image.astype(np.uint8)  # Ensure the image is a NumPy ndarray
        return noisy_image

################
# Load Dataset #
################
data = np.load(CONFIG['data_path'])
train_images, train_labels = data['train_images'], data['train_labels']
test_images, test_labels = data['test_images'], data['test_labels']

###################
# Train function  #
###################
#? To match the training scheme in MedMNIST.
def lr_lambda(epoch):
    initial_lr = 0.001  # Initial learning rate
    if epoch < 50:
        return initial_lr / initial_lr  # Learning rate remains 0.001
    elif epoch < 75:
        return 0.1 * initial_lr / initial_lr  # Delay learning rate to 0.0001 after 50 epochs
    else:
        return 0.01 * initial_lr / initial_lr  # Delay learning rate to 0.00001 after 75 epochs

def train(model, train_loader, task, epsilon_p=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_targets = []
    all_outputs = []
    
    # Track maximum gradient norm
    max_grad_norm = 0
    epoch_alpha_values = []

    if task == "multi-label, binary-class":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss() 
    criterion.to(CONFIG['device'])

    lr = 0.001
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = LambdaLR(optimizer, lr_lambda)

    for epoch in tqdm.tqdm(range(CONFIG['num_epochs'])):
        epoch_max_grad_norm = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.float().to(CONFIG['device']), targets.to(CONFIG['device'])
            batch_size = inputs.size(0)
            
            # Calculate per-sample gradients and find the maximum norm
            for i in range(batch_size):
                model.zero_grad()
                
                # Set model to eval mode temporarily for individual sample processing
                model.eval()  # This disables batch norm's training behavior
                
                # Forward pass for single sample
                output_i = model(inputs[i:i+1])
                
                if task == 'multi-label, binary-class':
                    target_i = targets[i:i+1].to(torch.float32)
                    loss_i = criterion(output_i, target_i)
                else:
                    # Use view(-1) to make target a 1D tensor of shape [1] instead of [1,1]
                    target_i = targets[i:i+1].view(-1).long()
                    loss_i = criterion(output_i, target_i)
                
                # Backward pass
                loss_i.backward()
                
                #? Get all grads to flatten
                grad_flat = []
                for param in model.parameters():
                    if param.grad is not None:
                        grad_flat.append(param.grad.flatten())

                if len(grad_flat) > 0:
                    all_grads = torch.cat(grad_flat) #! Concat all
                    grad_norm = all_grads.norm(2).item() #! Compute norm
                else:
                    grad_norm = 0.0

                epoch_max_grad_norm = max(epoch_max_grad_norm, grad_norm) #! For max of epochs
                
                # Set model back to train mode for batch training
                model.train()
            
            # Perform normal batch training
            optimizer.zero_grad()
            outputs = model(inputs)
            
            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32)
                loss = criterion(outputs, targets)
                
                # Calculate accuracy for multi-label
                predicted = (outputs > 0.5).float()
                correct += (predicted == targets).all(dim=1).sum().item()
            else:
                targets = targets.squeeze().long()
                loss = criterion(outputs, targets)
                
                # Calculate accuracy for standard classification
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
            
            total += targets.size(0)
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
            # Store targets and outputs for AUC calculation
            all_targets.extend(targets.cpu().numpy())
            all_outputs.extend(outputs.detach().cpu().numpy())
        
        max_grad_norm = max(max_grad_norm, epoch_max_grad_norm)
        epoch_alpha_values.append(epoch_max_grad_norm)
        scheduler.step()
        print(f"Epoch {epoch+1}/{CONFIG['num_epochs']} - Max Gradient Norm: {epoch_max_grad_norm:.4f}")

    print(f"Final maximum gradient norm (alpha): {max_grad_norm:.4f}")
    return model, max_grad_norm, epoch_alpha_values


########
# Main #
########
# Define transformation
data_transform = transforms.Compose([transforms.ToTensor()])

# Initialize dictionaries to store results
results = defaultdict(lambda: defaultdict(list))

epsilon_values = [0.1, 1.0, 10]

for epsilon_p in epsilon_values:
    print(f"Running for epsilon_p: {epsilon_p}")
            
    # Create datasets
    train_dataset = MNISTLaplacianDataset(train_images, train_labels, epsilon_p=epsilon_p, transform=data_transform)
    test_dataset = MedMNISTDataset(test_images, test_labels, transform=data_transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    # Initialize model
    model = ResNet18(CONFIG['num_channels'], CONFIG['num_classes']).to(CONFIG['device'])

    # Train and test the model
    train_model, alpha, epoch_alpha_values = train(model, train_loader, CONFIG['task'], epsilon_p)
    acc, auc, _ = test(train_model, test_loader, CONFIG['task'])
    
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC: {auc:.4f}")
    
    # Store results
    results[epsilon_p]['alpha'] = alpha
    results[epsilon_p]['epoch_alpha_values'] = epoch_alpha_values
    results[epsilon_p]['accuracy'] = acc
    results[epsilon_p]['auc'] = auc
    
    del model, train_model, train_dataset, test_dataset, train_loader, test_loader

# Plot alpha values over epochs for different epsilon values
plt.figure(figsize=(10, 6))
for epsilon_p in epsilon_values:
    plt.plot(range(1, CONFIG['num_epochs']+1), results[epsilon_p]['epoch_alpha_values'], label=f'ε={epsilon_p}')
plt.xlabel('Epoch')
plt.ylabel('Max Gradient Norm (α)')
plt.title('Maximum Gradient Norm per Epoch for Different Epsilon Values')
plt.legend()
plt.grid(True)
plt.savefig('alpha_vs_epoch.png')
plt.close()

# Plot final alpha vs epsilon
plt.figure(figsize=(8, 5))
alpha_values = [results[eps]['alpha'] for eps in epsilon_values]
plt.plot(epsilon_values, alpha_values, marker='o')
plt.xlabel('Epsilon (ε)')
plt.ylabel('Maximum Gradient Norm (α)')
plt.title('Relationship between Epsilon and Gradient Bound (α)')
plt.grid(True)
plt.savefig('alpha_vs_epsilon.png')
plt.close()

# Print a summary of results
print("\nSummary of Results:")
print("=" * 50)
print(f"{'Epsilon':^10}|{'Alpha':^15}|{'Accuracy':^15}|{'AUC':^15}")
print("-" * 50)
for eps in epsilon_values:
    print(f"{eps:^10}|{results[eps]['alpha']:^15.4f}|{results[eps]['accuracy']:^15.4f}|{results[eps]['auc']:^15.4f}")