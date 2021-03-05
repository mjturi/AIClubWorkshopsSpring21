---
layout: page
title: Object Detection
show_in_menu: false
disable_anchors: true
---

# Object Tracking Workshop
In this workshop we will be scripting a program that is able to track different objects. Last week we had our object [detection workshop](https://hectorenevarez.github.io/AIClubWorkshopsSpring21/workshop3/ObjectDetection). A big disadvantage to using an object detection model like YOLO is the computational power required to run the script. A viable solution is pairing object detection with object tracking. One could implement a script that runs the object detection every n seconds while the object tracker tracks those objects in between. For this workshop we will only be implementing the object tracker.

## OpenCV
We will be using OpenCV for this workshop in order to process our frames and visually display our results. More specifically we will be using the opencv-contrib version. Through [PyPi](https://pypi.org/) we use pip to install opencv however opencv-python and opencv-contrib-python are both unofficial versions of OpenCV. The contrib version is the repo that holds all new and experimental algorithms. The algorithm we will be using for this workshop is not available in the opencv-python version.

# Getting Started
For this weeks workshop, if you'd like to participate in the live coding session, you must have your environment set up. If your environment isn't set up you can follow the [setup instruction](https://hectorenevarez.github.io/AIClubWorkshopsSpring21/workshop4/settingup) for this weeks workshop.

## Project Structure
```bash
.
├── objectTracking.py
├── objectTrackingWorkshop.py
├── utils
│   ├── config.py
│   ├── FPS.py
│   └── init__.py
└── videos
    └── street.mp4
```

Above we can see the working directory

- objectTracker.py: This file contains the complete code for the workshop
- ObjectTrackingWorkshop.py: We will be working in this file
- utils:
    - config.py: A file with configuration settings for our workshop
    - FPS.py: This file contains an implementation of an fps calculator class
- videos: A folder that contains any videos we'll use for testing

## Imports
First we need to get all of our ```imports``` taken care of. At the top of our file we will see all the required imports:

```python
from utils.config import BOX_COLOR, LINE_THICKNESS, WIDTH
from utils.FPS import FPS
import argparse
import imutils
import cv2
```

As explained before, the utils packages are in our local directory and are just used for small functions we'll see throughout the code. As far as our other modules, let's see what they actually do:
- [argparse](https://docs.python.org/3/library/argparse.html): This module allows us to parse user command line arguments
- [Imutils](https://github.com/jrosebr1/imutils): This module provides convenient functions for OpenCV as some OpenCV functions can be a bit complicated
- [OpenCV](https://opencv.org/): As mentioned above we use this module is used for real-time computer vision applications

## Argument Parser
This section we use in order to parse any required arguments
```python
ap = argparse.ArgumentParser() # parse args
ap.add_argument("-v", "--video", required=True, help="path to video file")
args = vars(ap.parse_args())
```

Here we create an instance of an ```ArgumentParser``` object. We then add a required argument which requires the user to specify the path to a video feed that they would want our software to operate on. Lastly we grab all the arguments and store them inside our args variables as ```vars``` which stores objects inside a dictionary object.

```python
if args["video"].isdigit(): # check int for peripheral cam
    args["video"] = int(args["video"])
```

I also included the ability for a user to enter a numerical value for the argument. This is because in OpenCV you can use your webcam as a video stream. This is usually done by passing in the argument 0. 

## Initializing variables
Before we read in the frames, we need to initialize a couple of our variables.
```python
tracker = cv2.TrackerKCF_create()
```
Here we are creating an instance of our tracker object. We are using the Kernalized Correlation Filter(KCF) implemented in OpenCV. The tracker attempts to determine the motion of a set of points, given an initial set of points, by observing the direction of change in the next frame. If it identifies a confident change, the bounding box points will be updated. 

We're using this specific tracker because it is fast and accurate. A big downside is that it does not handle occlusion well, but for our implementation, we won't have to worry about the trackings robustness.

OpenCV implements this tracker from the [High-Speed Tracking with Kernalized Correlation Filters](https://www.robots.ox.ac.uk/~joao/publications/henriques_tpami2015.pdf) paper.

A simpler, yet still math heavy, breakdown can be found [here](https://cw.fel.cvut.cz/b172/courses/mpv/labs/4_tracking/4b_tracking_kcf#:~:text=Tracking%20with%20Correlation%20Filters%20(KCF),-An%20object%20to&text=When%20testing%2C%20the%20response%20of,adapts%20to%20moderate%20target%20changes.)

```python
fps = FPS()
```
Next we instanciate the the fps class. This class just displays the fps of the program while tracking in order to demonstrate how quick it actually runs.

```python
bbox = None
```

We also initialize the bounding box to none in order to assign the variable later. THe bounding box will end up being a list of the coordinate points around a person.

Lastly, before we get started tracking objects in frame, we need to use OpenCV in order to process every frame in the video. We do that by creating a ```VideoCapture``` object in OpenCV:
```python
cap = cv2.VideoCapture(args["video"])
```
This allows us to process the video. Here we pass in the video argument we require using the argparser library we previously implemented. We did this because it allows us to quickly change the video we want to process without directly changing the code.

## Iterating Through Every Frame
Now that we have all our variables set up we can begin to iterate through every frame. We use our ```VideoCapture``` object to pull a frame every iteration in an infinite while loop.
```pytohn
while True:
    _, frame = cap.read()
    if frame is None:
        print("No more frames detected, ending video...")
        break
```
We also added a small error checking portion to see if no frames are detected or if its the end of a video in order to gracefully exit.

## Capturing keys
This next section will be done first however, it should be placed at the end of your while loop. The full script will be shown at the bottom so reference that for a better idea of placement.

Our goal is to track an object every frame. Before we get into updating the tracker, we must first initialize it. We'll do that by having the user click a key in order to allow them to select their region of interest. Since we're not using an object detector, the user will manually draw a box around the object it wants to track

```python
cv2.imshow("Frame", frame)
key = cv2.waitKey(1) & 0xFF

if key == ord('s') and not bbox:
    bbox = cv2.selectROI("Frame", frame)
    tracker.init(frame, bbox)
    fps.start()
```

Here we printed out the frame using the ```cv2.imshow()``` function. This will just display the frame as well as assign the window a name.

We also are storing a key variable. we do that by using the ```waitKey()``` function to capture the character code of any key pressed. We then use a bit mask , ```0xFF``` in order to grab the ASCII representation of the key pressed, if any. 

We then compare the key pressed with ```ord('s')``` which grabs the unicode value of the ```s``` key. If our values match up and we haven't decalred a bounding box, then we enter the if statement. For this tracker implementation, we can only track one object at a time, which is why we have to make sure there is no other bounding box already declared.

We then initialize the bounding box value using the ```cv2.selectROI()``` function which will prompt the user, in the frame, to drag and create a bounding box around an object they want to track. This function will return the coordinates to the bounding box.

Finally we start the fps object to get the fps diplayed in the frame.

```python
if key == ord('c'):
    tracker = cv2.TrackerKCF_create()
    bbox = None

if key == ord('q'):
    print("Exiting video...")
    break
```

Afterwards, we also create a condition to allow the user to reset the tracker. Again, since we are only allowing for one tracker to be created, we allow the user to reset and track something else by pressing the ```c``` key.

We also have an exit condition that gracefully exits the loop whenever the user pressed ```q```.

## Updating the Bounding Box
This portion of code should fall above the section we just completed. Reference the entire code below if there is any confusion.

Now we have to update the tracker based on the bounding boxes we obtained in the previous step. First, we need to capture some data.
```python
frame = imutils.resize(frame, width=WIDTH) #resize frame for faster processing
```

We made our frame size smaller in order to process data quicker.

```python
if bbox is not None: # checking for bounding box
    (ret, box) = tracker.update(frame)

    fps.update()
    fps.stop()
```
We then check to see if we have a bounded box(The user selected an object) and update our tracker. Our ```tracker.update``` method takes in a frame and returns a return status(True or False) and the new bounding box. 

We then update our fps counter and use the stop function to get the time elapsed.

```python
if ret:
    (x, y, w, h) = [int(num) for num in box]
    cv2.rectangle(frame, (x, y), (x + w, y + h), BOX_COLOR, LINE_THICKNESS)
    frame = fps.fps_print(frame) # prints fps on screen
```

Next we check if the return was succesful. This ```ret``` variables was obtained from our tracker update and it stores whether the tracker was succesful or not. If so, we display our results.

Using [list comprehension](https://www.w3schools.com/python/python_lists_comprehension.asp), we grab the values from our box, store them in a list, and distribute them between four different variables.

We then use the ```cv2.rectangle()``` function to display our bounding box. 

Lastly, we print out the fps.

# Closing Files
Lastly we want to close any files, windows, and captures that we opened using the ```imshow()``` function and the ```VideoCapture``` object:
```python
cap.release()
cv2.destroyAllWindows()
```

# Entire Code
```python
from utils.config import BOX_COLOR, LINE_THICKNESS, WIDTH
from utils.FPS import FPS
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser() # parse args
ap.add_argument("-v", "--video", required=True, help="path to video file")
args = vars(ap.parse_args())

if args["video"].isdigit(): # check int for peripheral cam
    args["video"] = int(args["video"])

tracker = cv2.TrackerKCF_create() # Creating KCF tracker instance
fps = FPS()
bbox = None

cap = cv2.VideoCapture(args["video"])

while True:
    _, frame = cap.read()
    if frame is None:
        print("No more frames detected, ending video...")
        break

    frame = imutils.resize(frame, width=WIDTH) #resize frame for faster processing

    if bbox is not None: # checking for bounding box
        (ret, box) = tracker.update(frame)

        fps.update()
        fps.stop()

        if ret:
            (x, y, w, h) = [int(num) for num in box]
            print((x, y, w, h))
            cv2.rectangle(frame, (x, y), (x + w, y + h), BOX_COLOR, LINE_THICKNESS)
            frame = fps.fps_print(frame) # prints fps on screen

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s') and not bbox:
        bbox = cv2.selectROI("Frame", frame)
        tracker.init(frame, bbox)
        fps = FPS()
        fps.start()

    if key == ord('c'):
        tracker = cv2.TrackerKCF_create()
        bbox = None

    if key == ord('q'):
        print("Exiting video...")
        break


cap.release()
cv2.destroyAllWindows()
```