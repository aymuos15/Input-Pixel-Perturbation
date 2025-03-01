import numpy as np
from torch.utils.data import Dataset

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

#? Normal Test Dataset
class MedMNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        
        return image, label