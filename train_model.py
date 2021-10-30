from os import path
import numpy as np 
import pandas as pd 
import json 

from PIL import Image

import torch 
from torch.utils.data import DataLoader, Dataset, dataloader
from torch import nn 

from torchvision import transforms
import torchvision
class BrainDataset(Dataset):
    
    def __init__(self, settings:dict, train:bool=True, transform:transforms=None) -> None:
        
        if train:
            self.path_to_data = settings["file"]["train_data"]
            self.data_list = list(pd.read_csv(self.path_to_data))
        else:
            self.path_to_data = settings["file"]["test_data"]
            self.data_list = list(pd.read_csv(self.path_to_data))            
        self.train_status = train
        self.transform = transform
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index:int) -> dict:
        path_to_img = self.data_list[index]
        img = Image.open(path_to_img + ".tif")
        target = transforms.PILToTensor()(Image.open(path_to_img + "_mask.tif"))
        
        if self.transform:
            img = self.transform(img)
            
        return img, target

def main():
    
    # Load settings
    with open("settings.json","r") as inFile:
        settings = json.load(inFile)
        
    # Set random seed and device
    np.random.seed(1984)
    device = 'cuda:0' if torch.cuda.is_available() else "cpu"
    
    # Create the transform
    transform = transforms.Compose([transforms.ToTensor(), 
        transforms.Normalize((0.0919, 0.0833, 0.0875), (0.1354, 0.1238, 0.1293))])
    
    # Crate the dataset
    train_dataset = BrainDataset(settings, train=True, transform=transform)
    test_dataset = BrainDataset(settings, train=False, transform=transform)
    
    # Create dataloader 
    dataloader_args = {"batch_size":settings["param"]["batch_size"],
                       "shuffle":True}
    train_dataloader = DataLoader(train_dataset, **dataloader_args)
    test_dataloader = DataLoader(test_dataset, **dataloader_args)
    
    # Load model 
    model = torchvision.models.segmentation.fcn_resnet101(pretrained=False, num_classes=1)
    model.to(device)
    
    # Create loss and optimizer 
    loss_func = nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr = settings['param']['learning_rate'])
    
    # Initialize 
    loss_train, loss_test, acc_train, acc_test = [], [], [], []
    
    # Train model 
    for _ in range(settings["param"]["n_epochs"]):
        model.train()
        loss_e, acc_e = [], []
        for img, target in train_dataloader:
            
            img, target = img.to(device), target.to(device)
            
            # Forward pass 
            out = model(img)
            out = out['out']
            
            # Compute loss 
            loss = loss_func(out.data, target)
            
            # Append loss and accuracy 
            loss_e += [loss.cpu().detach().numpy()]
            # acc_e += [compute_acc(out, target)]
            
            # Compute derivatives
            optimizer.zero_grad()
            loss.backward()
            
            # Update weights through back-propagation
            optimizer.step()
            
        # Append to global list 
        loss_train.append(np.mean(loss_e)); acc_train.append(np.mean(acc_e))
            
        # Verbose 
        print("Training Accuracy:" + str(loss_train[-1]))    
        print("Training Loss:" + str(acc_train[-1]))
            
            
            
    
    

if __name__ == "__main__":
    main()
    