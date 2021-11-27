The project was started by having a look on the complete project and what is required to be done in order to achieve success.

I used the Workspace/VM to perform the project

I started by analysis the tfrecords in the training_and_validation directory.

1) Exploratory Data Analysis
===========================

Two approaches I found helpful
a) to access each exmaple in the tfrecord and use the feature to read the encoded_image, bb, labels
   This approach was skipped as I had to give a specific tfrecord and get the data from inside and I would not be able to ask for batches. 
b) to use the get_dataset interface to get the dataset from all the tfrecords file and access them.
   I decided finally to go with this solution and I was able to get the required data and plot it as well with less commands and effort

Then I wanted to have a look on the images to make a good analysis on the dataset so I created getting_images_from_dataset.py
to automatically take number of batches and store the images in the /home/workspace/Photos folder.

Update ------> Class distribution
I have created a bar diagram that shows the amount of the 3 different classes in each dataset (train, val and test) by taking 500 samples
and as expected, there are not cyclist labels found in the test_dataset --> suggesstion is to update the testdataset to guarantee the performance of the model.

But for the other datasets, I believe the random split is not a bad choice.
Continue with the same chosen strategy.

2) Analyzing the Dataset
========================

From the study I noticed:
a) Different kind of vehicles
b) Pedestrians in different locations and sides of the human body
c) Different kind of lights (day and night)
d) Weather condition (sunny, clear, rain, fog)
e) Different neighborhoods (compounds, cities, highways)

There is no pattern found for the images, I will go with the random split

Note: Photos are deleted so I can save space on the VM

3) Create the training - validation splits
==========================================

Performed 80%-20% random split between train-valid dataset respectively
as per the previous analysis

4) Edit the config file
=======================

Config file settings were updated
train dataset
validation dataset
label_map
The pretrained model

this was done by running:
python edit_config.py --train_dir /home/workspace/data/train/ --eval_dir /home/workspace/data/val/ --batch_size 2 --checkpoint /home/workspace/experiments/pretrained_model/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 --label_map /home/workspace/experiments/label_map.pbtxt

Note: config file was further updated in the training phase.

5) Training and Evaluation
==========================

a) Copy pipeline_new.config to /home/workspace/experiments/reference
b) Training process
by running: python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config

I am thankful that in this project I had th eopportunity to train for more than 20 hours and had to explore many things.
While training and live tracking, I have aquired a lot of knowledge and tips on how to improve the training model.

I spent time working on
a) Augmentations: Random crop - Horizontal flop - Blackspots - Brightness - JPEG quality. Adding augmentations also doesnt guarantee better better performance.
b) Batch sizes: increasing and decreasing the size has an effect on the training processes. Also increasing did not improve the results.
c) Optimizer: momentum optimizer - Adams. Changing the optimizer also has an influence on the behaviour. Further studies will be done to know the exact change in behaviour and how to deal with it.
d) Momentum optimizer value: A balanced value had to be found to reach optimal results
e) Learning rate: cosine change. I liked using the it to see what happens while increasing and decreasing the learning rate, and because of this I realized that a smaller anngle was necessary to have a better model, but not so small as this has negative influence on the results.
f) Training steps: Increasing the total number of steps was not so helpful as the model was always saturating around 2k. And defenitly decreeasing it is not required.

Most of the tests I ran, I kept pictures for the total losses that can be found in /home/workspace/TrialResults. This directory contains only trials and not related to the final model.


After many trials, I decided on the below model model.
a) AcceptedResult
Training loss ended with 0.79 (the range of it was acceptable)
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.149
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.304
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.130
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.052
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.397
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.456
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.041
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.159
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.224
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.118
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.514
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.560
INFO:tensorflow:Eval metrics at step 2500
I1126 00:10:29.632929 140586117482240 model_lib_v2.py:988] Eval metrics at step 2500
INFO:tensorflow:    + DetectionBoxes_Precision/mAP: 0.149047
I1126 00:10:29.645383 140586117482240 model_lib_v2.py:991]  + DetectionBoxes_Precision/mAP: 0.149047
INFO:tensorflow:    + DetectionBoxes_Precision/mAP@.50IOU: 0.304091
I1126 00:10:29.647256 140586117482240 model_lib_v2.py:991]  + DetectionBoxes_Precision/mAP@.50IOU: 0.304091
INFO:tensorflow:    + DetectionBoxes_Precision/mAP@.75IOU: 0.130439
I1126 00:10:29.648963 140586117482240 model_lib_v2.py:991]  + DetectionBoxes_Precision/mAP@.75IOU: 0.130439
INFO:tensorflow:    + DetectionBoxes_Precision/mAP (small): 0.051714
I1126 00:10:29.650597 140586117482240 model_lib_v2.py:991]  + DetectionBoxes_Precision/mAP (small): 0.051714
INFO:tensorflow:    + DetectionBoxes_Precision/mAP (medium): 0.397433
I1126 00:10:29.652192 140586117482240 model_lib_v2.py:991]  + DetectionBoxes_Precision/mAP (medium): 0.397433
INFO:tensorflow:    + DetectionBoxes_Precision/mAP (large): 0.455557
I1126 00:10:29.653978 140586117482240 model_lib_v2.py:991]  + DetectionBoxes_Precision/mAP (large): 0.455557
INFO:tensorflow:    + DetectionBoxes_Recall/AR@1: 0.041131
I1126 00:10:29.655566 140586117482240 model_lib_v2.py:991]  + DetectionBoxes_Recall/AR@1: 0.041131
INFO:tensorflow:    + DetectionBoxes_Recall/AR@10: 0.159393
I1126 00:10:29.657060 140586117482240 model_lib_v2.py:991]  + DetectionBoxes_Recall/AR@10: 0.159393
INFO:tensorflow:    + DetectionBoxes_Recall/AR@100: 0.224490
I1126 00:10:29.658565 140586117482240 model_lib_v2.py:991]  + DetectionBoxes_Recall/AR@100: 0.224490
INFO:tensorflow:    + DetectionBoxes_Recall/AR@100 (small): 0.117880
I1126 00:10:29.660113 140586117482240 model_lib_v2.py:991]  + DetectionBoxes_Recall/AR@100 (small): 0.117880
INFO:tensorflow:    + DetectionBoxes_Recall/AR@100 (medium): 0.513622
I1126 00:10:29.661654 140586117482240 model_lib_v2.py:991]  + DetectionBoxes_Recall/AR@100 (medium): 0.513622
INFO:tensorflow:    + DetectionBoxes_Recall/AR@100 (large): 0.560028
I1126 00:10:29.663208 140586117482240 model_lib_v2.py:991]  + DetectionBoxes_Recall/AR@100 (large): 0.560028
INFO:tensorflow:    + Loss/localization_loss: 0.376012
I1126 00:10:29.664662 140586117482240 model_lib_v2.py:991]  + Loss/localization_loss: 0.376012
INFO:tensorflow:    + Loss/classification_loss: 0.268087
I1126 00:10:29.666023 140586117482240 model_lib_v2.py:991]  + Loss/classification_loss: 0.268087
INFO:tensorflow:    + Loss/regularization_loss: 0.244616
I1126 00:10:29.667254 140586117482240 model_lib_v2.py:991]  + Loss/regularization_loss: 0.244616
INFO:tensorflow:    + Loss/total_loss: 0.888714
I1126 00:10:29.668511 140586117482240 model_lib_v2.py:991]  + Loss/total_loss: 0.888714

Analysis:
Training and validation loss are not so different. But both numbers need to be decreased by improving the model. Please refer to FinalResults directory"
I was surprised that the precision and recall are low. Please refer to FinalResults directory"
The model does not recognize far vehicles. Noticed by generating the gifs
The model did not recognize parked vehicles in dark areas (can be improved by augmentation brightness by increasing the delta value) Noticed by generating the gifs
I believe also adding an augmentation regarding blue or image quality may improve the behaviour. Noticed by generating the gifs
I tried to improve them but did not get much time. Noticed by generating the gifs


I also did not like the fact that the test dataset did not include cyclist detection. I should have added extra tfrecord to ensure that the mdoel can capture that.

Suggestion--> learn more about the effects of augmentations, optimizers and learning rates. I think these topics need to be more studied, by this I can have better starting point and save myself alot of time.
But thanks to trying I was able to understand a lot of things now and definetly l learned something that I am so proud of.

Note: 
Improvements can be further done, but due to the VM time limitations I was not able udpate the model more than that.
Recall needs to be improved -> Augmentations needs update.

6) Augmentation
===============

The  augmantations that supported the efficiency of the model were the horizontal flip, random grey effect and random brightness effect. Example can be found in the Augmentation directory.
Other augmentations like black patches and random crop did not really assist the model to get better results.

6) Git repo was created after finishing the project
===================================================
a) Creation of local repo.
b) Creation of repo on GitHub.
c) Push from local to GitHub.
d) Further changes have been done to enhance the model.