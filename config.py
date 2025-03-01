import torch

# import medmnist #? for google colaboratory
from medmnist import INFO

HOME = '/home/localssk23/'
# HOME = '/root/' #! For Google Colab

dataset = 'pneumoniamnist'

# #######################################################################
# #! Just to download the data initially
# info = INFO[dataset]
# DataClass = getattr(medmnist, info['python_class'])
# train_dataset = DataClass(split='train', transform=None, download=True)
# #######################################################################

CONFIG = {
   "batch_size": 2048,
   "num_epochs": 3,

   "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
   
   "data_path": f'{HOME}.medmnist/{dataset}.npz',
   "result_path": f'{HOME}input_pixel_perturbation/results/',
   "normal_result_path": f'{HOME}input_pixel_perturbation/results_normal/',

   "num_folds": 10
}

data_flag = CONFIG['data_path'].split('/')[-1].split('.')[0]
info = INFO[data_flag]

device = CONFIG['device']

CONFIG['num_classes'] = len(info['label'])
CONFIG['num_channels'] = info['n_channels']
CONFIG['task'] = info['task']