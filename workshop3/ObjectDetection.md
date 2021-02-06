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

### 
First we need to get all of our ```imports``` taken care of. At the top of our file we will see all the required imports:

```python
import numpy as np
import argparse
import imutils
import cv2
```
