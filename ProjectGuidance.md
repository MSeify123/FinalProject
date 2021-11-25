The project was started by having a look on the complete project and what is required to be done in order to achieve success.

I am workong on the Workspace/VM so I am not using Github or any sort of configuation management what so ever.

I started by analysin the tfrecords in the training_and_validation directory.

1) Eploratory Data Analysis
===========================

two approaches I reached
a) to access each exmaple in the tfrecord and use the feature to read the encoded_image, bb, labels 
b) to use the get_dataset interface to get the dataset from all the tfrecords file and access them.
   I decided finally to go with this solution and I was able to get the required data and plot it as well

Then I wanted to have a look on the images to make a good analysis on the dataset so I created getting_images_from_dataset.py
to automatically take number of batches and store the images in the /home/workspace/Photos folder.

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

Most of the tests I ran, I kept pictures for the total losses that can be found in /home/workspace/TrialResults

After many trials, I decided to accept 2 models.
a) AcceptedResultsTmp
DONE (t=0.64s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.136
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.264
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.121
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.047
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.342
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.428
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.038
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.149
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.211
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.112
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.470
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.532
INFO:tensorflow:Eval metrics at step 2500
I1123 13:07:47.519288 139623775110912 model_lib_v2.py:988] Eval metrics at step 2500
INFO:tensorflow:    + DetectionBoxes_Precision/mAP: 0.135513
I1123 13:07:47.533574 139623775110912 model_lib_v2.py:991]  + DetectionBoxes_Precision/mAP: 0.135513
INFO:tensorflow:    + DetectionBoxes_Precision/mAP@.50IOU: 0.264056
I1123 13:07:47.535836 139623775110912 model_lib_v2.py:991]  + DetectionBoxes_Precision/mAP@.50IOU: 0.264056
INFO:tensorflow:    + DetectionBoxes_Precision/mAP@.75IOU: 0.121073
I1123 13:07:47.537673 139623775110912 model_lib_v2.py:991]  + DetectionBoxes_Precision/mAP@.75IOU: 0.121073
INFO:tensorflow:    + DetectionBoxes_Precision/mAP (small): 0.046671
I1123 13:07:47.539527 139623775110912 model_lib_v2.py:991]  + DetectionBoxes_Precision/mAP (small): 0.046671
INFO:tensorflow:    + DetectionBoxes_Precision/mAP (medium): 0.341750
I1123 13:07:47.541332 139623775110912 model_lib_v2.py:991]  + DetectionBoxes_Precision/mAP (medium): 0.341750
INFO:tensorflow:    + DetectionBoxes_Precision/mAP (large): 0.427589
I1123 13:07:47.543306 139623775110912 model_lib_v2.py:991]  + DetectionBoxes_Precision/mAP (large): 0.427589
INFO:tensorflow:    + DetectionBoxes_Recall/AR@1: 0.037992
I1123 13:07:47.545319 139623775110912 model_lib_v2.py:991]  + DetectionBoxes_Recall/AR@1: 0.037992
INFO:tensorflow:    + DetectionBoxes_Recall/AR@10: 0.149194
I1123 13:07:47.547142 139623775110912 model_lib_v2.py:991]  + DetectionBoxes_Recall/AR@10: 0.149194
INFO:tensorflow:    + DetectionBoxes_Recall/AR@100: 0.210906
I1123 13:07:47.548939 139623775110912 model_lib_v2.py:991]  + DetectionBoxes_Recall/AR@100: 0.210906
INFO:tensorflow:    + DetectionBoxes_Recall/AR@100 (small): 0.111530
I1123 13:07:47.551043 139623775110912 model_lib_v2.py:991]  + DetectionBoxes_Recall/AR@100 (small): 0.111530
INFO:tensorflow:    + DetectionBoxes_Recall/AR@100 (medium): 0.470380
I1123 13:07:47.552891 139623775110912 model_lib_v2.py:991]  + DetectionBoxes_Recall/AR@100 (medium): 0.470380
INFO:tensorflow:    + DetectionBoxes_Recall/AR@100 (large): 0.532433
I1123 13:07:47.554951 139623775110912 model_lib_v2.py:991]  + DetectionBoxes_Recall/AR@100 (large): 0.532433
INFO:tensorflow:    + Loss/localization_loss: 0.359084
I1123 13:07:47.557072 139623775110912 model_lib_v2.py:991]  + Loss/localization_loss: 0.359084
INFO:tensorflow:    + Loss/classification_loss: 0.294418
I1123 13:07:47.559004 139623775110912 model_lib_v2.py:991]  + Loss/classification_loss: 0.294418
INFO:tensorflow:    + Loss/regularization_loss: 0.250800
I1123 13:07:47.560597 139623775110912 model_lib_v2.py:991]  + Loss/regularization_loss: 0.250800
INFO:tensorflow:    + Loss/total_loss: 0.904302
I1123 13:07:47.562100 139623775110912 model_lib_v2.py:991]  + Loss/total_loss: 0.904302
INFO:tensorflow:Waiting for new checkpoint at experiments/reference/

PDF of the results can be found in /home/workspace/AcceptedResultsTmp


b) AcceptedResultsFinal
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.185
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.320
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.162
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.049
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.550
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.900
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.039
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.142
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.211
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.086
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.570
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.900
INFO:tensorflow:Eval metrics at step 2500
I1124 22:54:11.929896 139671688492800 model_lib_v2.py:988] Eval metrics at step 2500
INFO:tensorflow:    + DetectionBoxes_Precision/mAP: 0.184766
I1124 22:54:11.932137 139671688492800 model_lib_v2.py:991]  + DetectionBoxes_Precision/mAP: 0.184766
INFO:tensorflow:    + DetectionBoxes_Precision/mAP@.50IOU: 0.320246
I1124 22:54:11.934330 139671688492800 model_lib_v2.py:991]  + DetectionBoxes_Precision/mAP@.50IOU: 0.320246
INFO:tensorflow:    + DetectionBoxes_Precision/mAP@.75IOU: 0.162106
I1124 22:54:11.936348 139671688492800 model_lib_v2.py:991]  + DetectionBoxes_Precision/mAP@.75IOU: 0.162106
INFO:tensorflow:    + DetectionBoxes_Precision/mAP (small): 0.049216
I1124 22:54:11.938138 139671688492800 model_lib_v2.py:991]  + DetectionBoxes_Precision/mAP (small): 0.049216
INFO:tensorflow:    + DetectionBoxes_Precision/mAP (medium): 0.550045
I1124 22:54:11.939918 139671688492800 model_lib_v2.py:991]  + DetectionBoxes_Precision/mAP (medium): 0.550045
INFO:tensorflow:    + DetectionBoxes_Precision/mAP (large): 0.900000
I1124 22:54:11.941783 139671688492800 model_lib_v2.py:991]  + DetectionBoxes_Precision/mAP (large): 0.900000
INFO:tensorflow:    + DetectionBoxes_Recall/AR@1: 0.039062
I1124 22:54:11.943619 139671688492800 model_lib_v2.py:991]  + DetectionBoxes_Recall/AR@1: 0.039062
INFO:tensorflow:    + DetectionBoxes_Recall/AR@10: 0.142187
I1124 22:54:11.945566 139671688492800 model_lib_v2.py:991]  + DetectionBoxes_Recall/AR@10: 0.142187
INFO:tensorflow:    + DetectionBoxes_Recall/AR@100: 0.210938
I1124 22:54:11.947520 139671688492800 model_lib_v2.py:991]  + DetectionBoxes_Recall/AR@100: 0.210938
INFO:tensorflow:    + DetectionBoxes_Recall/AR@100 (small): 0.085714
I1124 22:54:11.950490 139671688492800 model_lib_v2.py:991]  + DetectionBoxes_Recall/AR@100 (small): 0.085714
INFO:tensorflow:    + DetectionBoxes_Recall/AR@100 (medium): 0.570000
I1124 22:54:11.952423 139671688492800 model_lib_v2.py:991]  + DetectionBoxes_Recall/AR@100 (medium): 0.570000
INFO:tensorflow:    + DetectionBoxes_Recall/AR@100 (large): 0.900000
I1124 22:54:11.954146 139671688492800 model_lib_v2.py:991]  + DetectionBoxes_Recall/AR@100 (large): 0.900000
INFO:tensorflow:    + Loss/localization_loss: 0.175825
I1124 22:54:11.955793 139671688492800 model_lib_v2.py:991]  + Loss/localization_loss: 0.175825
INFO:tensorflow:    + Loss/classification_loss: 0.143795
I1124 22:54:11.957430 139671688492800 model_lib_v2.py:991]  + Loss/classification_loss: 0.143795
INFO:tensorflow:    + Loss/regularization_loss: 0.243642
I1124 22:54:11.959026 139671688492800 model_lib_v2.py:991]  + Loss/regularization_loss: 0.243642
INFO:tensorflow:    + Loss/total_loss: 0.563262
I1124 22:54:11.960629 139671688492800 model_lib_v2.py:991]  + Loss/total_loss: 0.563262

PDF of the results can be found in /home/workspace/AcceptedResultsFinal

This was the chosen one AcceptedResultsFinal
                        ====================

And this is the one I used to create the annimation.
Animation1: is the one requested
Animation2: to test the model further
Animation3: to test the model further

Analysis:
I was surprised that the precision and recall are low.
I tried to improve them but did not get much time.
This can also be noticed from the annimation.

Suggestion--> learn more about the effects of augmentations, optimizers and learning rates. I think these topics need to be more studied, by this I can have better starting point and save myself alot of time.
But thanks to trying I was able to understand a lot of things now and definetly l learned something that I am so proud of.

Note: 
Improvements can be further done, but due to the VM time limitations I was not able udpate the model more than that.
Recall needs to be improved -> Augmentations needs update.

6) Git repo was created after finishing the project
===================================================
a) Creation of local repo
b) Creation of repo on GitHub
c) Push from local to GitHub