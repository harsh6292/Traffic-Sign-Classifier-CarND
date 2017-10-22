**Traffic Sign Recognition**


[//]: # (Image References)

[image1]: ./sign_vs_count.png "Visualization"
[image2_1]: ./augmented_images/shift_img_original.png "Shift Original"
[image2_2]: ./augmented_images/shift_img_shifted.png "Shift replaced"
[image3_1]: ./augmented_images/rot_img_original.png "Rotation Original"
[image3_2]: ./augmented_images/rot_img_rotated.png "Rotation replaced"
[image3_3]: ./augmented_images/gamma_img_original.png "Gamma Original"
[image3_4]: ./augmented_images/gamma_img_replaced.png "Gamma replaced"

[image4]: ./sign_images_original/keep_left_original.png "Keep Left"
[image5]: ./sign_images_original/pedestrian_original.png "Pedestrian"
[image6]: ./sign_images_original/river_bank_original.png "River bank"
[image7]: ./sign_images_original/roundabout_original.png "Roundabout mandatory"
[image8]: ./sign_images_original/speed_limit_60_original.png "Speed limit 60"
[image9]: ./sign_images_original/stop_original.png "Stop"


---
### Dataset

I used the power of NumPy library to calculate the statistics for the dataset.

* The training set consists of 34799 images in form of pixel array.
* The validation set consists of 4410 sample images.
* The testing set consists of 12630 sample images.
* There are total of 43 classes/labels of traffic signs in the dataset.
* The shape of image to be processed is in form 32x32x3 where height and width of image is 32 pixels with 3 channels for RGB colors.


#### Visualization of the dataset.

The below graphs shows the graph depicting number of sample images per type of sign.

![alt text][image1]

Speed limit traffic signs have high number of image samples as compared to other signs.

### Data Pre-processing

For pre-processing and augmenting the dataset, I rotated, shifted and changed the brightness of images.

While pre-processing the dataset, I observed that some of the class labels have high number of training samples while many other classes had very low number of training samples. Due to this, labels with high samples were affecting the model and it was not able to generalize well.

To add more data to the data set, only class labels whose number of samples were less than 600 were added. First, the image was shifted by 5 pixels to the right direction and 5 pixels down. Here is the image before and after shifting.

![alt text][image2_1]
![alt text][image2_2]


Next I applied rotation to the image. Rotation was simple and images were rotated by 90 degrees only.

![alt text][image3_1]
![alt text][image3_2]

Next, the image brightness was changes by a factor of 1/2. This resulted in the following image.

![alt text][image3_3]
![alt text][image3_4]

The final training dataset after augmenting included 58016 images instead of earlier 34799 images.



### Model Architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							                |
|                   |                                                   |
| Convolution 5x5   | 1x1 stride, valid padding, outputs 28x28x96 	    |
| RELU					    |												                            |
| Convolution 1x1   | 1x1 stride, valid padding, outputs 28x28x64       |
| RELU              |                                                   |
| Max pooling	      | 2x2 stride,  outputs 14x14x64 			            	|
|                   |                                                   |
| Convolution 5x5   | 1x1 stride, valid padding, outputs 10x10x48 	    |
| RELU					    |												                            |
| Convolution 1x1   | 1x1 stride, valid padding, outputs 10x10x32       |
| RELU              |                                                   |
| Max pooling	      | 2x2 stride,  outputs 5x5x32   			            	|
|                   |                                                   |
| Convolution 1x1   | 1x1 stride, valid padding, outputs 5x5x24 	      |
| RELU					    |												                            |
| Convolution 1x1   | 1x1 stride, valid padding, outputs 5x5x16         |
| RELU              |                                                   |
|                   |                                                   |
| Flatten           | Input (5x5x16) flattened to output (400 nodes)    |
|                   |                                                   |
| Fully connected		| Input 400 nodes, outputs 200 nodes  							|
| RELU              |                                                   |
|                   |                                                   |
| Fully connected   | Input 200 nodes, outputs 100 nodes                |
| RELU              |                                                   |
|                   |                                                   |
| Fully connected   | Input 100 nodes, outputs 84 nodes                 |
| RELU              |                                                   |
|                   |                                                   |
| Fully connected   | Input 84 nodes, outputs 43 classes/labels         |


### Model Training

To train the model, I used batch size of 128 images per epoch. The small batch size was used to keep the memory consumption low during training.

Total number of epochs for which training continued was 25 epochs. The optimizer used to train the model is AdamOptimizer. The learning rate used at the start of first epoch is 0.0001. The optimizer was used as it tends to decrease its dependence on learning rate as the training progresses i.e. learning rate continues to decay with AdamOptimizer.


### Approach

I started building the architecture based off LeNet mode. Since LeNet model was used to train on images of digits, this problem is also similar to identifying lines inside an image.

However, since LeNet was built only for recognizing numbers with only 10 classes, the traffic signs dataset did not perform well the LeNet architecture.

To achieve better accuracy on the traffic sign dataset with LeNet, I added one more convolution layers to all the existing convolution layers. These were just 1x1 convolution layers to extract more high-level features from the previous convolution layer. So, for example, if LeNet had 1st convolution layer, the traffic sign model had LeNet's first convolution layer and then another 1x1 convolution layer before applying max pooling to these two convolution layers.

Even after increasing the number of layers, the accuracy was just reaching maximum of .93 at certain epochs and started oscillating between .88 and .93. From analyzing the model, I found the depth of the convolution layers was not enough for the model to generalize well. So, I increased the depth of first convolution layer to 96 and then started decreasing depths in next layers to reach the same number of nodes at flatten layer as LeNet (400). I also added one more fully connected layer at the end.


My final model results were:
* training set accuracy of -
* validation set accuracy of - 0.959 at end of epoch 20
* test set accuracy of - 0.95

From the above accuracy of validation and test accuracy results, it seems clear that model is performing well and generalized as both accuracies are very close to each other.

### Test a Model on New Images

Here are five German traffic signs that I found on the web:

![alt text][image6] ![alt text][image8] ![alt text][image4]
![alt text][image5] ![alt text][image7] ![alt text][image9]

The first image might be difficult to classify because this image is not present in the dataset.

The second image be difficult to classify because there are multiple signs with different speed limits which all look the same.

The third and fourth image (left arrow and pedestrian) are not so difficult to classify as they are fairly recognizable and distinguishable.

The roundabout sign (5th image) is a little difficult to classify as it matches many other signs which have round shape in it.

The stop sign is not difficult to classify due to its octagonal shape and stop text.


#### Model Predictions on new Images

Here are the results of the prediction:

| Image			             |     Prediction	        		  	|
|:----------------------:|:------------------------------:|
| No Pedestrian Sign     | No Pedestrian sign      			  |
| Keep left sign     		 | Keep left sign 								|
| Roundabout mandatory	 | Speed limit (100 km/hr)				|
| Speed limit (60 km/hr) | Speed limit (50 km/hr)					|
| Stop sign			         | Stop sign      							  |
| River bank sign			   | Bicycle crossing      					|


The model was able to correctly guess 3 of the 6 traffic signs, which gives an accuracy of 50%. The model does not perform well as it performed on the test set. This might be due to first, the river bank image is not in the dataset. Secondly, the model was not able to distinguish between different speed limit sign at such low resolutions of images.


For all the images below, model is quite certain that the output it predicts is actually that label. There is a huge difference between the top softmax probability and the second softmax probability.

###### No pedestrian sign
| Probability         	|     Prediction	        					|
|:---------------------:|:------------------------------:|
| 1.o         			    | No pedestrian sign   									|
| 7.41e-09     				  | Right-of-way at the next intersection 					|
| 1.82e-10					    | General caution											|
| 1.2e-19	      			  | Road narrows on the right					 				|
| 5.4e-20				        | Traffic signals      							|


###### Keep left sign

| Probability         	|     Prediction	        					|
|:---------------------:|:------------------------------:|
| 0.99         			| Keep left sign   									|
| 3.07e-05     				| Turn right ahead 										|
| 1.89e-10					| Roundabout mandatory											|
| 8.94e-13	      			| Turn left ahead					 				|
| 6.15e-15				    | Speed limit (100 km/hr)      							|


###### Roundabout mandatory sign

| Probability         	|     Prediction	        					|
|:---------------------:|:------------------------------:|
| .99         			| Speed limit (100 km/hr)    									|
| 7.79e-05     				| Go straight or left 										|
| 6.45e-07					| Roundabout mandatory										|
| 1.20e-10	      			| Speed limit (120 km/hr)d					 				|
| 3.64e-11				    | Speed limit (80 km/hr)      							|


###### Speed limit (60 km/hr) sign

| Probability         	|     Prediction	        					|
|:---------------------:|:------------------------------:|
| .99         			| Speed limit (50 km/hr)  									|
| 1.20e-06     				| Speed limit (60 km/hr) 										|
| 2.06e-08					| Speed limit (80 km/hr)											|
| 1.03e-08	      			| Speed limit (30 km/hr)					 				|
| 5.25e-13				    | Road work      							|


###### Stop sign

| Probability         	|     Prediction	        					|
|:---------------------:|:------------------------------:|
| .949         			| Stop sign   									|
| 5.04e-02     				| Yield  										|
| 5.19e-12					| Speed limit (80 km/hr)											|
| 3.12e-12      			| Speed limit (30 km/hr)				 				|
| 1.11e-12				    | No vehicles      							|


###### River bank sign

| Probability         	|     Prediction	        					|
|:---------------------:|:------------------------------:|
| .99         			| Bicycles crossing   									|
| 5.29e-04     				| Bumpy road 										|
| 2.70e-06					| Speed limit (60 km/hr										|
| 4.28e-07	      			| Slippery road					 				|
| 1.72e-07				    | No passing for vehicles over 3.5 metric tons      	|
