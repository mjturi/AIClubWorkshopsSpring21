---
layout: page
title: Optical Character Recognition
show_in_menu: false
disable_anchors: true
---

# What is OCR?
Optical Character Recognition(OCR) that is typically used to detect text from within images. You might have used applications such as scanning a credit card or check with your phone in order for the software to extract your information. The technology used for that is called OCR. 

<p align="center">
  <img width="460" height="300" src="https://miro.medium.com/max/737/1*FCjWFJVYOl1phvmWKJCO2w.png">
</p>

In this workshop we will be taking a look at OCR and implementing our own software that uses a preexisting OCR library to read the text off any image that we'd like.

## EasyOCR
The library we'll be taking a look at is [easyOCR](https://www.jaided.ai/easyocr/). EasyOCR is a python package that easily performs optical character recognition. It can be implemented quickly and simply. 

Other OCR libraries usually have different dependencies that make them hard to work with, but as the name suggests easyOCR is easy to implement and use. The library supports over 80 languages and is robust even with noisier images.

It was buily using PyTorch which enables us to use CUDA-capable GPU'd in order to speed up the detection tremendously.

At the moment easyOCR mainly supports OCR friendly text; text that is easily legible and not handwritten. The library is quickly expanding and plans to eventually support handwritten detection.

For more information on how the model was trained check out their [github](https://github.com/JaidedAI/EasyOCR)

# Getting Started
For this weeks live coding session we will be implementing easyOCR and using openCV in order to analyze the models results. Let's get started!<br><br>
[Live Coding](https://colab.research.google.com/drive/1IOO-bZY6mB2YUHp6XdtgOMJi07Zb94qF?usp=sharing) <br>
[Complete Code](https://colab.research.google.com/drive/1MwbFNKYAt7Mq-B65gqbLn1djx4hc4xRC?usp=sharing)<br><br>

## Imports
As usual we will begin by loading in our imports. Unlike other libraries built in to Google Colab, in order to use easyOCR we first have to install the library using the python package manager.
```python
!pip install easyocr
```

We use the exclamation point in Google Colab in order to let the cell know that we are running a command and not python code.

```python
from easyocr import Reader
import matplotlib.pyplot as plt
import urllib
import cv2
```

Then we import the rest of our packages.
- easyocr: Used for basic OCR functions we will be implementing
- matplotlib: Used for plotting our images
- urllib: Allows us to grab random pictures from the internet and turn them into interpretable images
- cv2: computer vision library we use to draw on our images

A key thing to note is that we only import Reader from easyocr. Reader is the main class we will be using in order grab text from images which is why it's the only class/function we need to import from easyocr.

## Loading in Images
First we have to pull an image from the web and convert it to an image
```python
def url_to_image(url):
  resp = urllib.request.urlretrieve(url, 'img_from_web.jpg')
  image = cv2.imread('img_from_web.jpg')
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  return image
```
We create this helper function to do that work for us. We first send a request to retrieve the image from the url and save it to our local storage. After, we read it using opencv and store it as an image. Finally we convert the colorspace of our image. Opencv reads in images as BGR and we want to see them as RGB so we make the appropriate conversion

```python
url = "https://www.bbvausa.com/content/dam/bbva/usa/en/photos/checking-and-savings/clearpoints-card-gateway-sm.png"
img = url_to_image(url)

plt.imshow(img)
plt.show()
```

We then store our URL as a string variable and invoke our url_to_image function in order to retrieve the image. Afterwards, we plot this image to make sure everything looks correct

## Extracting Text From Our Image
Now that we have our image in place, we use easyOCR to extract text from our image.
```python
reader = Reader(['en'], gpu=True)
results = reader.readtext(img)
```

We create an instance of a Reader object. We first specify what language we want to detect in. Multiple languages can be passed in this list. We also specify we'd like to use GPU for the detection in order to speed up the process. finally we use our reader to read the text. If we print the read data we get the following:
```
([[62, 49], [313, 49], [313, 139], [62, 139]], 'BBVA', 0.9981149435043335)
```
We have 3 main pieces of data that are being read: bounding box location, text, and probability. we then use this information to add text to our image.

## Processing Our Image
We'll want to put our results in a for loop in order to make sure we process all the information obtained.
```python
for (bbox, text, prob) in results:
  # Grab bounding box values
  (tl, tr, br, bl) = bbox
  tl = (int(tl[0]), int(tl[1])) # top left
  tr = (int(tr[0]), int(tr[1])) # top right
  br = (int(br[0]), int(br[1])) # bottom right
  bl = (int(bl[0]), int(bl[1])) # bottom left

  cv2.rectangle(img, tl, br, (0, 255, 0), 2)
  cv2.putText(img, text, (tl[0], tl[1]),
  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
```
We grab the four corners detected based on the (x,y) coordinates provided by easyOCR's detection. We then take this information and draw a rectangle around the text and add the text to the image.

```python
plt.imshow(img)
plt.show()

img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imwrite("OCR'ed_image.jpg", img) # automatically converts from bgr to rgb
```
Finally we display our image and save the image as a file for any uses we might need.