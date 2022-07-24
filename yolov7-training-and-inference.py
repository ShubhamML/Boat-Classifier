# %% [markdown]
# # YOLO (YOU ONLY LOOK ONCE)
# 
# ![image.png](attachment:5accfacc-5988-40b3-8182-f9c8f15847c9.png)

# %% [markdown]
# ### Just a few weeks ago, YOLO v7 [(research paper)](https://arxiv.org/abs/2207.02696) came into the limelight by beating all the existing object detection models to date. Anyone who has worked in Object detection has heard about YOLO. It’s been here for a while now, and to date, we have seen a lot of YOLO versions. YOLO is not a single architecture but a flexible research framework written in low-level languages. The framework has three main components: the head, neck, and backbone. Different sets of components and architecture are associated with the above-mentioned three components giving rise to different YOLO versions

# %% [markdown]
# # What is object detection?

# %% [markdown]
# ![image.png](attachment:b7842338-140e-4d71-bec4-4b22fa0fd525.png)
# 
# 
# 
# ### Object Detection is an advanced form of image classification where a neural network predicts objects in an image and points them out in the form of bounding boxes.Object detection thus refers to the detection and localization of objects in an image that belong to a predefined set of classes.Tasks like detection, recognition, or localization find widespread applicability in real-world scenarios, making object detection (also referred to as object recognition) a very important subdomain of Computer Vision 

# %% [markdown]
# # YOLO Architecture

# %% [markdown]
# ### Inspired by the GoogleNet architecture, YOLO’s architecture has a total of 24 convolutional layers with 2 fully connected layers at the end. 
# 
# ![image.png](attachment:e6281e50-a3d9-4822-b01b-6c7bf71040da.png)
# 
# 
# ## **Here's a timeline showcasing YOLO's development in recent years.**
# 
# 
# ![image.png](attachment:d61b7228-c395-4b0e-9efe-17fb49463b27.png)

# %% [markdown]
# # Performance 
# 
# ![image.png](attachment:7220100d-0a84-44aa-bc84-c507ccc564c0.png)

# %% [markdown]
# **We can see that the new yolov7 is performing 120% faster than the yolov5 version which was one of the best object detection models.**

# %% [markdown]
# # Training on Yolov7

# %% [markdown]
# **Firstly we will clone the git repo from the official yolov7**

# %% [code] {"execution":{"iopub.status.busy":"2022-07-24T06:07:29.300297Z","iopub.execute_input":"2022-07-24T06:07:29.300788Z","iopub.status.idle":"2022-07-24T06:07:30.067430Z","shell.execute_reply.started":"2022-07-24T06:07:29.300750Z","shell.execute_reply":"2022-07-24T06:07:30.066109Z"}}
!git clone https://github.com/WongKinYiu/yolov7.git
%cd ./yolov7

# %% [markdown]
# ### After cloning you just have to convert your data into yolov7 format , you can check the format from the github repo or simply upload your dataset on roboflow and convert it to v7 format. After that you just have to run the training script

# %% [markdown]
# # train p5 models
# !python train.py --workers 8 --device 0 --batch-size 32 --data data/coco.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights '' --name yolov7 --hyp data/hyp.scratch.p5.yaml
# 
# # train p6 models
# !python train_aux.py --workers 8 --device 0 --batch-size 16 --data data/coco.yaml --img 1280 1280 --cfg cfg/training/yolov7-w6.yaml --weights '' --name yolov7-w6 --hyp data/hyp.scratch.p6.yaml

# %% [markdown]
# **This will train the model on your dataset and save the weights and metrics in 'runs' folder , you can then perform inference from your custom models**

# %% [markdown]
# # Inference

# %% [code] {"execution":{"iopub.status.busy":"2022-07-24T06:07:35.051541Z","iopub.execute_input":"2022-07-24T06:07:35.051934Z","iopub.status.idle":"2022-07-24T06:07:54.391317Z","shell.execute_reply.started":"2022-07-24T06:07:35.051902Z","shell.execute_reply":"2022-07-24T06:07:54.389867Z"}}
# On image:
!python detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --source inference/images/horses.jpg

# %% [markdown]
# ![image.png](attachment:df6acbf7-d49b-4cbf-a728-c748b4d49a06.png)

# %% [markdown]
# **This is just a sample inference , you can provide your custom weights and perform inference on your custom images , its that easy!**

# %% [markdown]
# ### YOLO provided a super fast and accurate object detection algorithm that revolutionized computer vision research related to object detection.With over 6 versions (3 official) and cited more than 16 thousand times,  YOLO has evolved tremendously ever since it was first proposed in 2015.YOLO has large-scale applicability with thousands of use cases, particularly for autonomous driving, vehicle detection, and intelligent video analytics.

# %% [markdown]
# **Thank You for reading the notebook and please give it an upvote if you liked it :) .**

# %% [code]
