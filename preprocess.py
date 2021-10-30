import numpy as np 
import pandas as pd 
import json
from scipy.sparse.construct import random 
from sklearn.model_selection import train_test_split
import os 
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import torch, torchvision
from torch import tensor
from torch.utils.data import DataLoader, Dataset

class normBrainDataset (Dataset):
    
    def __init__(self, settings, train:bool=True, transform:torchvision.transforms=None) -> None:
        
        self.train_data_path = settings["file"]["train_data"]
        self.train_data = list(pd.read_csv(self.train_data_path, header=None, sep = "\n")[0])            
        self.test_data_path = settings["file"]["test_data"]
        self.test_data = list(pd.read_csv(self.test_data_path, header=None, sep="\n")[0])
        self.data_paths = self.train_data + self.test_data
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.data_paths)
    
    def __getitem__(self, index:int) -> tensor:
        image = Image.open(self.data_paths[index] + ".tif")
        
        if self.transform:
            image = self.transform(image)
        
        return image

def create_train_test(settings:dict) -> None:
    """create_train_test [Create train and test csv file]

    Args:
        settings (dict): [settings dictionary]
    """
    
    # Parse patients 
    pat_list = os.listdir(settings["dir"]["Data"])
    
    # Train test split
    train_subjects, test_subjects = train_test_split(pat_list, test_size=0.2, shuffle=True)
    
    # Create list of files
    train_data, test_data = [],[]
    for subj in train_subjects:
        subj_path = os.path.join(settings["dir"]["Data"], subj)
        train_data += [os.path.join(subj_path, img.split(".")[0]) for img in os.listdir(subj_path) if "mask" not in img]
    for subj in test_subjects:
        subj_path = os.path.join(settings["dir"]["Data"], subj)
        test_data += [os.path.join(subj_path, img.split(".")[0]) for img in os.listdir(subj_path) if "mask" not in img]
        
    
    # Write subject csv files 
    with open(settings["file"]["train_subjects"], "w") as outFile:
        for pat in train_subjects:
            outFile.write("%s\n" % pat)
    with open(settings["file"]["test_subjects"], "w") as outFile:
        for pat in test_subjects:
            outFile.write("%s\n" % pat)
            
    # Write train data and test csv files
    with open(settings["file"]["train_data"], "w") as outFile:
        for pat in train_data:
            outFile.write("%s\n" % pat)        
    with open(settings["file"]["test_data"], "w") as outFile:
        for pat in test_data:
            outFile.write("%s\n" % pat)
            
def compute_mean_std(settings) -> None:
    
    # Create transform 
    transform = transforms.transforms.ToTensor()
    
    # Crate dataset and dataloader
    brain_dataset = normBrainDataset(settings, transform=transform)
    brain_dataloader = DataLoader(brain_dataset)
    
    # Initialize
    psum    = torch.tensor([0.0, 0.0, 0.0]).unsqueeze(0)
    psum_sq = torch.tensor([0.0, 0.0, 0.0]).unsqueeze(0)
    
    # Compute sum 
    for img in tqdm(brain_dataloader):
        psum += img.sum(dim=[2,3])
        psum_sq += (img**2).sum(dim=[2,3])
        
    # Compute means and std
    pixel_count = len(brain_dataset)*(img.shape[-1]**2)
    total_mean = psum / pixel_count
    total_std = torch.sqrt((psum_sq/pixel_count) - total_mean**2)
    
    # Print 
    print("Mean: " + str(total_mean))
    print("Standard deviation: " + str(total_std))
    
    # Mean: tensor([[0.0919, 0.0833, 0.0875]])
    # Standard deviation: tensor([[0.1354, 0.1238, 0.1293]])

def main():
    
    # Load settings 
    with open("settings.json", "r") as inFile:
        settings = json.load(inFile)
        
    # Set random seed
    np.random.seed(1984)
        
    # Create train and test dataset
    create_train_test(settings)
    
    # Compute mean and std for each channel 
    compute_mean_std(settings)
    
    
    
if __name__ == "__main__":
    main()