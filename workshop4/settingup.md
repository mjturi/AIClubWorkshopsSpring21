---
layout: page
title: Workshop Setup
show_in_menu: false
disable_anchors: true
---

# Object Tracking Workshop Preparation
For this workshop we will have a live coding session. In order to participate in that portion, you'll need to have your python environment all set up with the proper packages as
well as download any necessary files. Follow the steps below in order to get started.<br>

First you'll need to have ```Python``` downloaded on your computer. Follow the video linked below in order to get started<br>
**Note** The version of python doesn't matter too much. As long as you are installing anything ```>= Python 3.5```
- [Windows Python Installation](https://www.youtube.com/watch?v=UvcQlPZ8ecA&t=47s)
- [Mac Python Installation](https://www.youtube.com/watch?v=TgA4ObrowRg)

You'll also need an IDE or text editor to help you edit your code. For that I recommend ```Visual Studio Code```
- [Windows Visual Studio Code Installation](https://www.youtube.com/watch?v=MlIzFUI1QGA)
- [Mac Visual Studio Code Installation](https://www.youtube.com/watch?v=tCfbi5PF1y0)

Once you have Python as well as an enviroment to work on your code, you'll need a couple packages. Using ```pip (package management system)``` we're going to install 2 packages.
In any terminal/Command prompt run the following commands
```
pip install opencv-contrib-python
pip install imutils
```

**NOTE** It is very important that you have the correct opencv version downloaded. Please install the ```opencv-contrib-python``` library mentioned above and remove any other opencv installation you might have. This can be done through pip by uninstalling:
```
pip uninstall opencv-python
```

This should install the packages locally so that you have access to them. Your way of installing packages might be different, you might have to use pip3 instead, depending on your installion process

To make sure the packages have been downloaded correctly, in your terminal / Command prompt run the following command
```
pip list
```
This should list all your installed packages. There you should see ```imutils```, and ```opencv-contrib-python``` listed

Finally, you'll need to download the required files. From my dropbox download the following folder
- [Object Detection Workshop](https://www.dropbox.com/sh/r8jlo0kk6znm22h/AAArpoPadBKbdVFicOLHo5CYa?dl=0)

Once it is downloaded unzip the folder and open it on your code editor and navigate to the directory where the ```objectTrackerWorkshop.py``` file is located. In your terminal run the following command:
```
python objectTrackingWorkshop.py -v 0
```
If the log shows that eeverything is all set up then you should be ready, otherwise you might have to trouble shoot