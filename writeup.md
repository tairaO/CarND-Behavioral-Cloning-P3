# **Behavioral Cloning**


### Behavioral Cloning Project

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./fig_submit/model.png "Model Architecture"
[image2]: ./fig_submit/centerdriving_example.jpg "Center-driving Image"
[image3]: ./fig_submit/recovery_1.jpg "Recovery Image"
[image4]: ./fig_submit/recovery_2.jpg "Recovery Image"
[image5]: ./fig_submit/recovery_3.jpg "Recovery Image"
[image6]: ./fig_submit/normal_image.png "Normal Image"
[image7]: ./fig_submit/flipped_image.png "Flipped Image"
[image8]: ./fig_submit/measurement_data.png "All of the steering control data"
[image9]: ./fig_submit/0.25_0.2_after_getting_data.png "Steering control data after removing straight driving"
[image10]: ./fig_submit/nvidia_convdropout1.png "Loss and validation loss of the models versus epochs"
[image11]: ./fig_submit/nvidia_convdropout2.png "Loss and validation loss of the models versus epochs"
[image12]: ./fig_submit/raw_measurements.png "Raw measurements histogram"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup.md summarizing the results
* run.mp4 the movie of my model

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes and 3x3 filter sizes and depths between 24 and 64 (model.py lines 129-143)

The model includes RELU layers to introduce nonlinearity (code line 129-137), and the data is normalized in the model using a Keras lambda layer (code line 125) and cropped upper 70 and lower 25 rows in images (code line 126).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers every after convolutional layer in order to reduce overfitting (model.py lines 130,132,134,136,138).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 70). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was the default value and not tuned manually (model.py line 146).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use a convolution neural network model similar to NVIDIA architecture. I thought this model might be appropriate because it does not use max-pooling and it has enough depth to extract lanes' edge information. (Actually, I first used LeNet architecture but it didn't produce good result) I think, in this project, we humans are using lanes edges to drive appropriate position in lanes. Normally, first convolutional layer extracts edges. Also, according to [Zeiler](https://www.youtube.com/watch?v=ghEmQSxT6tw&t=18s), max-pooling somewhat forms rotational invariant. But in this project, rotational invariant is not useful. So I guessed the architecture that uses max-pooling is not good in this project because it lost edge's angle information.

My first step was to use a convolution neural network model exact same as NVIDIA architecture.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model so that I could get higher validation rate.

Then I applied dropout layers every after convolutional layer.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track like the banks after bridge and  to improve the driving behavior in these cases, I gathered more recovery data (so I didn't change architecture).

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 129-143) is NVIDIA architecture, but every after convolution layer, I applied dropout with the keeping rate 0.5.

Here is a visualization of the architecture.

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps each on track one CCW and CW using center lane driving with mouse steering controller. Here is an example image of center lane driving. This:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from side of the lanes. These images show what a recovery looks like starting from side of the lanes. This decreased left-bias because normally CCW driving tend to gather left-steering, vice versa. Therefore, driving CW and CCW helps model being generalized:

![alt text][image3]
![alt text][image4]
![alt text][image5]

To augment the data set, first, I remove 75% of straight driving data because my gathered data has a lot of straight driving data and if I use it all of the data, my model would be biased to go straight. Here are the all of the steering control distribution and one after removing straight driving.

![alt text][image8]
![alt text][image9]




Second, I also flipped images and angles thinking that this would help the model generalize. For example, here is an image that has then been flipped. This is because my steering control has more data of left driving. It is shown in [the histogram below][image12].  Therefore, only normal images, it tends to go left. Flipping images helps improve this bias.:

![alt text][image6]
![alt text][image7]
![alt text][image12]


After the collection process, I had 15318 number of data points. I then preprocessed this data by normalizing and cropping upper 70 rows and lower 25 rows.


I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 30 as evidenced by the comparatively lower validation value and its stopping improving the validation rate. I used an adam optimizer so that manually training the learning rate wasn't necessary.

![alt text][image10]
![alt text][image11]
