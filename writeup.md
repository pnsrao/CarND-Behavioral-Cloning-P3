
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.
A python generator may be needed

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with two layers of 5x5 filter sizes and depths of 6 and 16. There is a MaxPooling layer after each convolutional layer. Subsequently there are two fully connected layers.

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. 
Finally there is a fully connected layer that outputs the steering angle. This is based on the Lenet architecture.

Details are in the file model.py between lines x and y.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

## #Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to LeNet. Since this task involves recognizing aspects of video images that aid driving, an architecture used in other image classification tasks was deemed suitable. I used a simplified version of Lenet because there are no requirements to recognize and classify complicated shapes. Instead the objective is to identify certain features (track boundaries, surface differences between the track and surrounding areas). And then use a fully connected layer to estimate the steering angle.

I first trained the model on the Udacity data. I split the data into training and validation sets. Initial training and validation errors seemed to be roughly similar. So I did not use any regularization techniques at this point and deferred this aspect for later.

The next step was to try the model within the driving simulator. The car was driving off road in quite a few places. Hence I used techniques of training data augmentation by using image flips as well as multiple camera images. These techniques reduced the off-road instances considerably but did not eliminate them completely. 

The next step was to explicitly collect recovery data for training. So I had training runs through the simulator where I positioned (but did not record) the car close to the edge of the road at a bad angle. Then I recorded the recovery part of the drive. This was done at many parts of Track 1

The final part involved an issue where the car drove into a side dirt track (after the bridge) and did not follow the road that curved left wards. Since this problem persisted even after repeated efforts to retrain at this location. I had to think of another way. Instead of processing all three color channels, I converted the images into the (H,S,V) domain and extracted the saturation (S) channel. These proves to be an effective discriminator for the road vs surrounding areas. This pre-processing steps helps in overcoming the issue and the car is able to navigate the entire track at this point.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. However the car was driving wobbly at times and the steering angles seemed to be noisy. I then increased the complexity of the network to the original Lenet. 
With the increased complexity of the network, I saw that the the training errors were noticeably smaller than validation errors, i.e., the model was overfitting the data. At this point, I added dropout layers after the first two fully connected layers.

#### 2. Final Model Architecture

My model consists of a convolution neural network with two layers of 5x5 filter sizes and depths of 6 and 16. There is a MaxPooling layer after each convolutional layer. Subsequently there are two fully connected layers with output sizes 120 and 84 respectively. There is a dropout layer with 50% dropout after each of these two fully connected layers.

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. 
Finally there is a fully connected layer that outputs the steering angle.

Details are in the file model.py between lines 58 and 72.


#### 3. Creation of the Training Set & Training Process

As described in previous sections, collection of training data and the training process proved to be the most important part of the  project.

I started by using the Udacity training set as a baseline for "typical" driving. Then I added a training set that consisted mainly of driving along curves of the road. Finally I added a recovery set that entirely consisted of the car recovering from bad positions.
Since the main objective is to avoid driving off road, the recovery data should be collected when the car is still within the road but is badly positioned.

As part of the training process, I did use techniques of data augmentation (flipping images and multiple cameras) and also used the pre-processing step of extracting the S-channel from the (H,S,V) decomposition. At the end, I removed the multiple cameras images since my recovery set was good enough for the purpose of keeping the car within the road. So I could rely only on the center camera.

#### 4. Driving on the track
Track 1 was used to test autonomous driving based on the model. The record of the autonomous driving lap is in the video.
