import argparse
import glob
import os
import random
import shutil

import numpy as np

from utils import get_module_logger

def create_dirs():
    # Directories to be created
    directory_train = "train"
    directory_val = "val"
    directory_test = "test"
  
    # Parent Directory path to those new directories
    parent_dir = "/home/workspace/data"
    
    # Create pathes to those directory
    path_train = os.path.join(parent_dir, directory_train)
    path_val= os.path.join(parent_dir, directory_val)
    path_test = os.path.join(parent_dir, directory_test)
    
    # Create the directories
    os.mkdir(path_train)
    os.mkdir(path_val)
    os.mkdir(path_test)
    
    

def split(data_dir):
    """
    Create three splits from the processed records. The files should be moved to new folders in the 
    same directory. This folder should be named train, val and test.

    args:
        - data_dir [str]: data directory, /home/workspace/data/waymo
    """
    
    # TODO: Split the data present in `/home/workspace/data/waymo/training_and_validation` into train and val sets.
    # You should move the files rather than copy because of space limitations in the workspace.
    
    # Create folders
    create_dirs()
    
        
    ############################################################### Split into 2 #########################################################################
    
    # Creation of destinations for the split
    destination_training_path = "/home/workspace/data/train"
    destination_validation_path = "/home/workspace/data/val"
    destination_testing_path = "/home/workspace/data/test"
    
    #There are 97 files, I will do the ratio as below
    # 80% Training_dataset   --> 75 .tfrecord files
    # 20% Validation_dataset --> 22 .tfrecord files
    
    #Move data_dir to train_and_validation directory
    data_dir_train_and_valid = os.path.join(data_dir, 'training_and_validation')
    data_dir_test = os.path.join(data_dir, 'test')
    
    # Loop over all .tfrecord files
    
    for i in range (0,97):
        
        #Get a random file from the directory
        filename = random.choice(os.listdir(data_dir_train_and_valid))
        #Prepare path to the filename
        Path_to_File = f"{data_dir_train_and_valid}/"+ filename
        
        #Move 80% to train
        if i < 75:           
            shutil.move(Path_to_File, destination_training_path)
        #Move 20% to valid   
        else:
            shutil.move(Path_to_File, destination_validation_path)
            
    #Just to move files from /data/waymo/test to data/test       
    for i in range (0,3):
        #Get a random file from the test directory
        filename = random.choice(os.listdir(data_dir_test))
        Path_to_File = f"{data_dir_test}/"+ filename
        shutil.move(Path_to_File, destination_testing_path)
        
    
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--data_dir', required=True,
                        help='data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.data_dir)
    