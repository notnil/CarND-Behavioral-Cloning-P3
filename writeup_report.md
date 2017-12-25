# **Behavioral Cloning** 

[//]: # (Image References)

[image1]: ./examples/image1.png "Center Driving"
[image2]: ./examples/image2.png "Reverse Driving"
[image3]: ./examples/image3.png "Recovery Driving"

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing (on macOS) 
```sh
docker run -it --rm -p 4567:4567 -v `pwd`:/src udacity/carnd-term1-starter-kit python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The CNN based model architecture is documented in the "Final Model Architecture" section. 

#### 2. Attempts to reduce overfitting in the model

Overfitting, in the network sense, was solved using dropout layers in the network architecture.  I also experienced overfitting in the sense that my training data didn’t fully represent a generalized environment of different car orientations, parts of the road, backgrounds, boarder types, etc.  This type of overfitting was combatted with augmenting training data and adding exceptional data (recovery lap) to the training set.

#### 3. Model parameter tuning

Parameter Tuning Techniques
- Adam optimizer to adapt learning rate dynamically 
- Dropout increased to reduce overfitting
- Left and Right camera steering offset to augment dataset
- Cropping values from images to reduce data size

#### 4. Appropriate training data

Training data was gathered to fully satisfy the needs of autonomous driving.  Distinct types of training data were:
- Counter clockwise center lane driving with keyboard input
- Counter clockwise center lane driving with mouse input
- Clockwise center lane driving with keyboard input
- Counter clockwise “recovery” driving with keyboard input

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

Since the input data was images, I elected to use a convolutional neural network based architecture.  When researching Keras, I stumbled across the [Keras Guide to Sequential Models](https://keras.io/getting-started/sequential-model-guide/).  Their VGG-like convent example was the basis for my architecture.  I enlarged the model beyond the example so that it was complex enough to represent the problem space.  

Since I employed dropout from the beginning of developing my architecture, I did not experience overfitting of my network to its training data.  I did however experience overfitting of my network to its training data compared to the entire solution space.  I combatted this problem with data acquisition and augmentation techniques detailed in a section below.

At the end of the process, the vehicle was able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 Image   					        |
| Lamda         		| Normalization and mean centering   			|
| Cropping         		| Cut off 70px from top and 50px from bottom   	| 
| Convolution Layer    	| filters=32 kernal=3x3 activation=relu 	    |
| Convolution Layer    	| filters=32 kernal=3x3 activation=relu 	    |
| Max pooling	      	| 2x2 pool size and stride 				        |
| Dropout				|												|
| Convolution Layer    	| filters=64 kernal=3x3 activation=relu 	    |
| Convolution Layer    	| filters=64 kernal=3x3 activation=relu 	    |
| Max pooling	      	| 2x2 pool size and stride 				        |
| Dropout				|												|
| Convolution Layer    	| filters=128 kernal=3x3 activation=relu 	    |
| Convolution Layer    	| filters=128 kernal=3x3 activation=relu 	    |
| Max pooling	      	| 2x2 pool size and stride 				        |
| Dropout				|												|
| Flatten				| 								                |
| Fully connected		| size=256 activation=relu        				|
| Dropout				|												|
| Fully connected		| size=256 activation=relu        				|
| Dropout				|												|
| Fully connected		| size=1 activation=None        				|

#### 3. Creation of the Training Set & Training Process

After a half lap of test driving, I started recording test data on track one.  During my first session I simply tried to stay inside the lines.  I didn't have a mouse at the time so I utilized the keyboard based driving method.  I successfully completed two laps without going outside the lane markers so I saved the training data and started working on my model.  Here is a picture of the kind of driving from the first training run.  

![alt text][image1]

At this point I was curious to see how a simple convolutional model would perform so I trained a network using only the data from two laps using only the center camera.  Unsurprisingly it didn't make it around the first turn.  Since the mean square error for training and validation wasn't improving with more epochs, I decided more data was necessary.  Firstly I drove a clockwise lap around the course which was opposite of the starting position.  Secondly I drove a "recovery" lap where I made a bunch of short recording returning to the track from off the road.  Here is an image showing the recovery process:

![alt text][image3]    

I retrained the model and to my surprised it performed only slight better than the first iteration.  Tired of using the simulator, I channeled my efforts into data augmentation.  To quickly double my dataset I flipped all the center images in the generator function.  Excited about the sudden enlargement of the dataset, I retrained the network and reran it in autonomous mode.  Disappointingly it didn't preform much better.

Next I utilized the left and right camera images in my training data.  I had been putting off this step because fine tuning the steering offset parameter seemed like a "guess and check" process.  My initial guess was a 0.4 offset.  While tripling the dataset, using the left and right camera images also made training three times as long.  Since additional epochs didn't improve the overall loss, I reduced the number of epochs to one.  After training the model, I reran the simulator in autonomous mode.  It completed the entire lap.  I was very surprised at the huge leap in performance (and that my first offset guess had been successful) and started thinking about why it worked so well.  My conclusion was that I had a "middle of the road" bias and collecting data from the sides (simulated using left and right cameras) created a more generalizable model.