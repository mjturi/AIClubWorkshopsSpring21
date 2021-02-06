---
layout: page
title: Object Detection
show_in_menu: false
disable_anchors: true
---

# Object Detection Workshop
In this workshop we will be scripting a program that uses the YOLOv3 object detection model to detect different objects in a frame. Specifically we will configure the software to detect only people.

## What is YOLO
YOLO (you only look once) is a state of the art object detector that is able to recognize 80 different objects in images and videos, but more importantly, it is super fast and accurate for what it is able to handle. We won't go into the specifics of how YOLO was trained and why it is so effective but you can find more information regarding it on the [YOLO website](https://pjreddie.com/darknet/yolo/). There is a YOLOv4 that has come out except, it can't be run on OpenCV.

## What is OpenCV
[OpenCV](https://opencv.org/opencv-ai-competition-2021/) is an open source computer vision and machine learning software library. The library provides thousands of optimized algorithms and it is a great tool for real time applications. I gave a workshop last semester going over OpenCV that you can check out [here](https://hectorenevarez.github.io/AIClubWorkshopsFall20/#workshop-6-computer-vision-1). Its ok if your not too familiar how OpenCV for todays workshop. I'll be explaining everything throughout the workshop.

# Getting Started
During last weeks meeting I mentioned that to participate in the live coding session you had to have everything setup; your python environment, a text editor/IDE, and the necessary files to follow along. If you don't have this setup you won't be able to participate in the live coding portion however you can try and quickly set everything up by following the [setup instructions](https://hectorenevarez.github.io/AIClubWorkshopsSpring21/workshop3/settingup).

Let's go ahead and get started. In our project directory, we will be working in the ```objectDetectionWorkshop.py``` file.

## Imports
First we need to get all of our ```imports``` taken care of. At the top of our file we will see all the required imports:

```python
import numpy as np
import argparse
import imutils
import cv2
```

Let's take alook at what these modules actually do:
- [NumPy](https://numpy.org/devdocs/user/whatisnumpy.html): This module is used for numerical computations in Python. It gives us access to an array object which has a ton of support for different mathematical operations and runs quick
- [argparse](https://docs.python.org/3/library/argparse.html): This module allows us to parse user command line arguments
- [Imutils](https://github.com/jrosebr1/imutils): This module provides convenient functions for OpenCV as some OpenCV functions can be a bit complicated
- [OpenCV](https://opencv.org/): As mentioned above we use this module is used for real-time computer vision applications

## Argument Parser
This section we use in order to parse our any required arguments
```python
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True, help="path to input video")
args = vars(ap.parse_args())
```

Here we create an instance of the ```ArgumentParser``` object. We then add a required argument which requires the user to specify the path to a video feed that they would want our software to operate on. Lastly we grab all the arguments and store them inside our args variables as ```vars``` which stores objects inside a dictionary object.

## Loading in the Model
Now we have to load in the YOLO model. First, we'll load in the YOLO model:
```python
net = cv2.dnn.readNetFromDarknet("models/yolov3.cfg", "models/yolov3.weights")
```

To load in the model we used OpenCV's DNN function. This function is able to interepret YOLO's weights and configuration. In Layman's terms, this loads in the YOLO object detection model

We also have to load in the class names for YOLO. As I had mentioned, yolo is able to detect up to up to 80 different objects. We want to get these object names in order to reference them later when we only detect for people.

```python
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
```

Here we get the unconnected layers from the model. To understand this concept you need to understand the structure of a Neural Network. If you don't understand the structure of a Neural Network its ok, we are essentially just getting the different labels for the objects it contains by grabbing the final layer of the neural network which typically has the output layer. The output layer determines what the model classified something as.

I also included, by accident, another form of getting the labels of detected objects:
```python
LABELS = open("models/coco.names").read().strip().split("\n")
```

Here we use a file that already has the names of the output layer. We won't be using this implementation, but just know it is an alternative

## Capturing Video
Lastly, before we get started detecting objects in frame, we need to use OpenCV in order to process every frame in the video. We do that by creating a ```VideoCapture``` object in OpenCV:
```python
cap = cv2.VideoCapture(args["video"])
```

This will go and allow us to process the video. Here we pass in the video argument we require using the argparser library we previously implemented. We did this because it allows us to quickly change the video we want to process without directly changing the code.

# Processing Every Frame
The way in which we detect objects on a video is by running YOLO on every frame and visually diplaying our results. In this section we will be working inside an infinite while loop. This will allow us to go frame by frame running our results.

As you can imagine this is computationally expensive. Running a model on every frame eats up a lot of memory. In this situation since we are processing our results using our CPU, it gets pretty expensive quick. There are more advanced methods to help with processing our results faster and getting a quick runtime speed but for the simplicity of this workshop, we will be keeping it like this

## Reading a Frame
Using the ```VideoCapture``` object we created, we'll extract the first frame:
```python
_, frame = cap.read()
```
Since ```cap.read()``` returns two values, we store the first value inside the underscore variable and the second inside the frame variable. We'll only be using the frame variable which is why we gave the other variable the name underscore. The frame variable holds an individual frame from a video.

Now we also want to extract data from the frame for future use:
```python
HEIGHT, WIDTH, CHANNEL = frame.shape
```

This just gives us some useful information from the frame like the shape and height.

## Displaying our frames
I tend to put this section at the bottom of the while loop, however I always include it first so we can see what we're working with.
```python
cv2.imshow("frame", frame)

if cv2.waitKey(1) & 0xFF == ord('q'):
    break
```

Here the function ```imshow()``` displays our frame in its own window. We give the frame a name and then we give the information needed to print the frame. Here we also specify an early quit statement. Here we are checking to see if the user presses "q" while in frame, the video will exit.

## Detecting Objects
Now we can finally get into the object detection portion. We'll use our YOLO model to extract information from the frame and determine what has been detected
```python
blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
layerOutputs = net.forward(ln)
```

First we create a blob object through OpenCV. Typically when you want to run a model, you preprocess your data so it matches what the model is looking for. In this case the ```blobFromImage``` function sharpens our image and performs more specific preprocessing techniques. Here's an example:
![Blob](https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.pyimagesearch.com%2F2017%2F11%2F06%2Fdeep-learning-opencvs-blobfromimage-works%2F&psig=AOvVaw2Ps9gmdchYJH7pyLuQkyaq&ust=1612729821200000&source=images&cd=vfe&ved=0CAIQjRxqFwoTCOjrgfSM1u4CFQAAAAAdAAAAABAD)
