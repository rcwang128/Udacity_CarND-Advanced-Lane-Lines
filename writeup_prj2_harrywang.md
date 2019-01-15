# **CarND-term1 Advance Lane Finding** 
# Project 2
# Harry Wang
---



Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.

* Apply a distortion correction to raw images.

* Use color transforms, gradients, etc., to create a thresholded binary image.

* Apply a perspective transform to rectify binary image ("birds-eye view").

* Detect lane pixels and fit to find the lane boundary.

* Determine the curvature of the lane and vehicle position with respect to center.

* Warp the detected lane boundaries back onto the original image.

* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

  

[//]: # "My Output Image/video Examples"

[image1]: ./output_images/camera_calibration.png	"Camera calibration"
[image2]: ./output_images/undistorted_camera.png	"Undistorted image"
[image3]: ./output_images/warp_camera.png "Top down warp image"
[image4]: ./output_images/combined_thresholding.png "Color and gradients thresholding"
[image5]: ./output_images/full_search_and_lane_fitting.png "Detected lane image"
[image6]: ./output_images/processed_image.png "Fully processed image example"
[video1]: ./output_videos/project.mp4 "Fully processed video example "



### Rubric](https://review.udacity.com/#!/rubrics/571/view) Points



---

### The code for entire project is contained in the IPython notebook located in "**./examples/example.ipynb**"

### All my data is on github: https://github.com/rcwang128/Udacity_CarND-Advanced-Lane-Lines



---

### Camera Calibration

#### Step 1. Calibration on test images.

`objpoints` contains the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  
`imgpoints` contains the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.
`objpoints` and `imgpoints` are used to compute the camera calibration and distortion coefficients through `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained below results.

![1547573090181](/home/harry/.config/Typora/typora-user-images/1547573090181.png)



---

### Pipeline (single images)

#### Step 2. Apply distortion correction to camera images.

I use the distortion coefficients to un-distort each camera images. Example output is showing below. 

![1547575459102](/home/harry/.config/Typora/typora-user-images/1547575459102.png)



#### Step 3. Perspective Transform.

The `img_warp()` function is created to perform a perspective transform. `src` and `dst` are practical source and destination points for image warping, which are fixed for this particular application. `cv2.warpPerspective()` function is used for perspective transofrm. An "birds-eye view" example picture can is showing below. 
```python
src = np.float32([[30,720], [500,480], [800,480], [1250,720]])
dst = np.float32([[30,720], [0,0], [1280,0], [1250,720]])
```

![1547576206693](/home/harry/.config/Typora/typora-user-images/1547576206693.png)



#### Step 4. Image Thresholding.

This step is to create a binary image for lane detection by combing different thresholding methods. I applied color space transforms as well as gradient thresholding to meet the purpose. 

For color thresholding, I've tried different color spaces and ended up using h and l channels.  An example output is showing as below.

![1547576881390](/home/harry/.config/Typora/typora-user-images/1547576881390.png)

For gradient thresholding, I used sobel, gradient magnitude and direction all three types of thresholding methods to get a reliable lane picture.

![1547577344874](/home/harry/.config/Typora/typora-user-images/1547577344874.png)

Different thresholding methods have their own pro and cons when dealing with different situations like shadows or color changes on paved roads. Therefore, I combined all these types to output a most confident binary images with lane clearly detected. The function`combined_thresh()` is defined to server the purpose. Example output images are showing below.
```python
def combined_thresh(img):
    # Color thresholding
    hls_binary = hls_select(img, thresh=(100, 255))
    luv_binary = luv_select(img, thresh=(220, 255))
    # Gradient thresholding
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=1, abs_thresh=(20, 120))
    grady = abs_sobel_thresh(img, orient='y', sobel_kernel=7, abs_thresh=(10, 120))
    mag_binary = mag_thresh(img, sobel_kernel=7, mag_thresh=(30, 100))
    dir_binary = dir_threshold(img, sobel_kernel=15, dir_thresh=(0.7, 1.3))
    # Combined thresholding
    combined = np.zeros_like(dir_binary)*255
    combined[(hls_binary == 1 ) | (luv_binary == 1) \
           	| (gradx == 1) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    
    return combined
```

![1547578069402](/home/harry/.config/Typora/typora-user-images/1547578069402.png)



#### Step 5. Lane Detection.

This is the main step where I detect left and right lanes on the binary image I generated with thresholding. Basically I have two major functions namely `find_lane_n_fit_poly()` and `search_around_n_fit_poly()`.

`find_lane_n_fit_poly()` function takes a binary image and perform a full search using sliding window method. It firstly find two highest peaks from image histogram determining where the lane lines are, and then use sliding windows moving upward in the image to determine where the lane lines go. After that, it identifies the nonzero pixels within the window to locate the lines. All line pixel positions are then extracted, and are used for line fitting. A second order polynomial line fitting method is used to fit curved lanes. This function returns bunch of line parameters for further calculation needs. An example picture is showing below.

![1547579694301](/home/harry/.config/Typora/typora-user-images/1547579694301.png)

`search_around_n_fit_poly()` function on the other hand, is created for the case where a lane has been detected successfully in previous frame. It takes previous lane information and only search the nonzero pixels within a certain given margin/window. After pixel positions are extracted, the rest fitting method is similar. An example picture is showing below where the red fitting lines are mimicked from previous frame.

![1547579943850](/home/harry/.config/Typora/typora-user-images/1547579943850.png)



#### Step 6. Determine the curvature and vehicle position.

Most line parameters are already extracted during previous step. `measure_curvature()` function here is just to calculate curvature. The horizontal and vertical distance is scaled based on my warp images.
```python
# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/900 # meters per pixel in x dimension
```



#### Step 7. Draw the polygon.

`fillpoly()` function collects all points for left and right fitting lines and draw the polygon between them.



#### Step 8. Warp back and print info.

Finally, I created my last function for processing single images as `img_process()`. Within this function, I followed above mentioned steps to process an entire image or frame. It returns a fully processed image and curvature as well as center position information for printing. Example pictures are showing below.
```python
def img_process(img):
    # Undistort image after camera calibration
    undist_img = undistort(img, objpoints, imgpoints)
    # Warp img and Generate top down view
    top_down, perspective_M = image_warp(undist_img, src, dst)
    # Apply combined thresholding
    combined_img = combined_thresh(top_down)
    # Calculate fitting lines
    left_fitx, right_fitx, ploty, left_fit, right_fit, left_fit_cr, right_fit_cr, center, out_img = find_lane_n_fit_poly(combined_img)
    # Fill the polygon
    filled_img = fillpoly(combined_img, left_fitx, right_fitx, ploty, src, dst)
    # Warp back the polygon filled image
    M_inv = cv2.getPerspectiveTransform(dst, src)
    filled_img = cv2.warpPerspective(filled_img, M_inv, (img.shape[1], img.shape[0]))
    # Add it to original image
    processed_img = cv2.addWeighted(img, 1, filled_img, 0.5, 0)
    # Measure curvature
    left_curverad, right_curverad = measure_curvature(left_fit_cr, right_fit_cr, ploty)
    curvature = int((left_curverad + right_curverad)/2)
    
    return processed_img, curvature, center
```

![1547582891554](/home/harry/.config/Typora/typora-user-images/1547582891554.png)



---

### Pipeline (video)

#### Step 9. Process the video.

Processing video is very similar as processing images since each frame can be treated as a single image. However, for better efficiency and smooth continuity, I have implemented a simple algorithm to separate the lane detection into two functions.

As previously mentioned, `search_around_n_fit_poly()` function is efficient, and being called more frequently as the lanes are detected for most of the time. Every time when lanes are detected, the fitting line parameters are saved and delivered to the function for quick lane detection in next frame. This helps to smooth line fitting frame by frame.

Under some conditions for example when a lane is not being detected in previous frame, a full lane search is necessary and thus `find_lane_n_fit_poly()` function has to be applied. To better deal with difficult road environment, I also implemented a `sanity_check()` function, which checks if the detected left and right lanes are very off, or the calculated vehicle position is unreasonable. This sanity check is there to be part of conditions for "search from prior" calls.

Here's a [link to my video result](./output_videos/project.mp4)



---

### Discussion

#### Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My algorithm works very stable for most part of the project video. It tends to be a little wobbly when in the shadow (about 42 sec in video), but is quickly adjusted back to normal. However, my algorithm performs much worse in challenge videos. 

One of the reasons it is trying to fail under tough environment (e.g. shadow) may due to my lane detection algorithm. During rapid color change periods, my thresholding method, or lane fitting method is not good enough to detect lanes. Another reason could be my sanity check algorithm is not able to filter out the incorrect detection and thus stick to it for a frame or two until full search condition is met.

Gladly my algorithm still does the job to adjust itself by performing a full frame search when needed. If I have more time, I would try to improve my thresholding algorithm, which is the fundamental of lane detection. And then I should design a better filters and sanity check functions for better consistency.

