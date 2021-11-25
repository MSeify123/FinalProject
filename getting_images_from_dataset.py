import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.patches import Rectangle


import tensorflow.compat.v1 as tf
from waymo_open_dataset import dataset_pb2 as open_dataset

from utils import get_dataset
from object_detection.utils import dataset_util, label_map_util


    

def display_instances(counter, instance):
    """
    This function takes a batch from the dataset and display the image with 
    the associated bounding boxes.
    """
    
    # Variables
    name    = instance['filename']
    img     = instance['image'].numpy()
    img_shape = img.shape
    bboxes   = instance['groundtruth_boxes'].numpy()
    classes = instance['groundtruth_classes'].numpy()
    
   
    _, ax = plt.subplots(1,figsize=(20, 10))
    # color mapping of classes
    colormap = {1: [1, 0, 0], 2: [0, 0, 1], 4: [0, 1, 0]}
        
    # Get the images and store them in /home/workspace/Photos direcotry
    '''
    # Get images for cylcists
    if 4 in classes: 
        imgplot = plt.imshow(img)
        image_path = "Photos_cyclist"
        plt.imsave(f"{image_path}/image"+str(counter)+".png",img)
    '''
    imgplot = plt.imshow(img)
    image_path = "Photos"
    plt.imsave(f"{image_path}/image"+str(counter)+".png",img)


dataset = get_dataset("/home/workspace/data/waymo/training_and_validation/*.tfrecord")
counter =0
for batch in dataset.shuffle(500):
    counter+=1
    display_instances(counter,batch)