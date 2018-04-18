# Kaggle_DataScienceBowl2018

test5a: A variation on Unet that contains Gated Refinement Units and Gated Units with depths of 2 and average pooling and max pooling. The contour is two erosions from the actual contour. The model took as input the image and predicted the mask and the "contour". The loss was bce_dice_loss and weighting was 70-30. 

test5b: A variation on Unet that contains Gated Refinement Units and Gated Units with depths of 3 and average pooling and max pooling. Extra images in nuclei folder are used and a generator is used for training, so no contours are used. The loss function is bce_dice_loss.

test5c: A variation on Unet that contains Gated Refinement Units and Gated Units with depths of 3 and average pooling and max pooling. The contour is two erosions from the actual contour. The model took as input the image and predicted the mask and the "contour". The loss was bce_dice_loss and weighting was 70-30.

test6: A variation on Unet that contains Gated Refinement Units and Gated Units with depths of 2 and average pooling and max pooling. Contours are not used and the generator function is used for training. The loss was bce_dice_loss.

test6b: A variation on Unet that contains Gated Refinement Units and Gated Units with depths of 3 and average pooling and max pooling. Contours are not used and the generator function is used for training. The loss was bce_dice_loss.

test6b3: A variation on Unet that contains Gated Refinement Units and Gated Units with depths of 4 and average pooling and max pooling. Contours are not used and the generator function is used for training. The loss was bce_dice_loss.

test7b: A variation on Unet that contains Gated Refinement Units and Gated Units with depths of 2 and average pooling and max pooling. Contours and bounding boxes are used. The loss functions for the mask, contour, and bounding box are bce_dice_loss, bce_dice_loss, and mean_squared_error (for the location of the center, the height and width of the box). There is no weighting of the loss functions.

test7b2: A variation on Unet that contains Gated Refinement Units and Gated Units with depths of 2 and average pooling and max pooling. Inputs are the extracted from various points in the ResNet50 pretrained model and the parameters are kept constant. Contours and bounding boxes are used. The centers of the boxes and the dimensions of the boxes are not treated separately. There are no class weights. The loss functions for the mask, contour, and bounding box center and dimensiosn are bce_dice_loss, bce_dice_loss, and mean_squared_error. There is no weighting of the loss functions.

test7c: A variation on Unet that contains Gated Refinement Units and Gated Units with depths of 2 and average pooling and max pooling. Inputs are the extracted from various points in the ResNet50 pretrained model and the parameters are kept constant. Contours and bounding boxes are used. The centers of the boxes and the dimensions of the boxes are treated separately. The class weights were set to class_weight={0:1,1:1000}. The loss functions for the mask, contour, and bounding box center, and bounding box dimensiosn are bce_dice_loss, bce_dice_loss, bce_dice_loss, and mean_squared_error. There is no weighting of the loss functions.

test7e: A variation on Unet that contains Gated Refinement Units and Gated Units with depths of 2 and average pooling and max pooling. Contours are used. Additionally, the model tries to predict where there is overlap between two cells. This is gathered by dilating an individual mask and checking if it overlaps with any other masks. If so, it is labeled as an overlap. These overlaps are then subtracted from the mask image during post processing. The loss functions for the mask, contour, and overlaps are bce_dice_loss, bce_dice_loss, and bce_dice_loss. There is no weighting of the loss functions.

Post processing:
A watershed method was used for separating overlapping cells. A gray image of the cells, after subtracting the overlap lines from test7e, is thresholded at .3 (pixel values from 0 to 1). Then a rank.median funciton was used with a disk(2) in order to denoise. Then the markers were generated using rank.gradient with disk(3) when less than 10, followed by ndi.label. These are used to generate gradients using a disk(2). Finally watershed was used. Then, each label is checked, by looking at labels in decreasing order (ignoring the largest one because it is background). Then it looks at when the gray values are greater than 0.196, removing smal objects less than 21, and filling in holes. 
