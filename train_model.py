import numpy as np 
import pandas as pd 
import json 



def main():
    
    # Load settings
    with open("settings.json","r") as inFile:
        settings = json.load(inFile)
        
    # Set random seed 
    np.random.seed(1984)
    
    # Crate the dataset and dataloader

if __name__ == "__main__":
    main()
    