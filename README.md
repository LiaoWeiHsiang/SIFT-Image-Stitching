# SIFT-Image-Stitching

<!-- ## Abstract  -->

We implement SIFT to find the correspondes between two image and stitch them. 
<!-- ![](https://i.imgur.com/t5GQzh6.jpg) -->
![](document/parrington.jpg)

![](document/lab.jpg)

## SIFT (Scale Invariant Feature Transform), P.S. Take the bamboo fox image as example
### Scale-space extrema detection
In this part, we need to find the keypoints which exist in various scale-space. The scale space of an image is produced from the convolution of a Gaussian kernel(Blurring) at different scales with the input image. Scale-space is separated into octaves and the number of octaves and scale depends on the size of the original image. So we generate several octaves of the original image. Each octave’s image size is half the previous one. Within an octave, images are progressively blurred using the Gaussian Blur operator.
#### Image1
![](document/Scale-space-1.jpg)
#### Image2
![](document/Scale-space-2.jpg)


<!-- <img style="float: right;" src="https://i.imgur.com/hsbePXY.jpg" width=50% height=50%> -->
<!-- ![](https://i.imgur.com/hsbePXY.jpg) -->
### DOG(Difference of Gaussian kernel)
This process is done for different octaves of the image in the Gaussian Pyramid. It is represented in below image:
![](document/dog-1.png)

#### Image1
![](document/dog-2.jpg)

#### Image2
![](document/dog-3.jpg)


### Keypoint localization
Every pixel in an image is compared with its 8 neighbors as well as 9 pixels in the next scale and 9 pixels in previous scales. This way, a total of 26 checks are made. If it is a local extrema, it is a potential keypoint. It basically means that keypoint is best represented in that scale. 
Once a keypoint candidate has been found by comparing a pixel to its neighbors, the next step is to perform a detailed fit to the nearby data for location, scale, and ratio of principal curvatures. This information allows points to be rejected that have low contrast (and are therefore sensitive to noise) or are poorly localized along an edge.
Taylor series expansion of scale space is used to get a more accurate location of extrema, and if the intensity at this extrema is less than a threshold value (0.03 as per the paper), it is rejected. DoG has a higher response for edges, so edges also need to be removed. A 2x2 Hessian matrix (H) is used to compute the principal curvature.
#### Image1
![](document/Keypoints-1.jpg)
#### Image2
![](document/Keypoints-2.jpg)

### Orientation assignment

Now we have scale invariance. The next thing is to assign an orientation to each keypoint to make it rotation invariance.

A neighborhood is taken around the keypoint location depending on the scale, and the gradient magnitude and direction is calculated in that region. An orientation histogram with 36 bins covering 360 degrees is created. Let's say the gradient direction at a certain point (in the “orientation collection region”) is 18.759 degrees, then it will go into the 10–19-degree bin. And the “amount” that is added to the bin is proportional to the magnitude of the gradient at that point. Once you’ve done this for all pixels around the keypoint, the histogram will have a peak at some point.

![](document/Orientation.jpg)

![](document/Orientation assignment.PNG)

The highest peak in the histogram is taken and any peak above 80% of it is also considered to calculate the orientation. It creates keypoints with same location and scale, but different directions. It contributes to the stability of matching.

### Keypoint descriptor
![](document/Keypoint_descriptor.png)

### Keypoint Matching!
![](document/Keypoint_Matching.jpg)

