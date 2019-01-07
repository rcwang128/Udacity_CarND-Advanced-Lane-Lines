import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#%matplotlib inline

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
#objp = np.array([objp])

image_count = 0

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('../camera_cal/calibration*.jpg')

# Step through the list and search for chessboard corners
for frame in images:
    #img = cv2.imread(fname)
    img = mpimg.imread(frame)
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    gray_shape = gray.shape[::-1]
    
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

    # If found, add object points, image points
    if ret == True:
        #corners = np.array([[corner for [corner] in corners]])
        print(image_count, len(corners[0]))
        
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
        #plt.subplot(121);plt.imshow(img)
        #cv2.imshow('img',img)
        #cv2.waitKey(500)
        
        # Perform a calibration
        #ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objp, corners, gray_shape, None, None)
        
        # Generate an un distorted image
        #undist_img = cv2.undistort(img, mtx, dist, None, mtx)
        #plt.subplot(122);plt.imshow(undist_img)
        
        image_count +=1
        
        # Plot the result
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        #f.tight_layout()
        ax1.imshow(mpimg.imread(frame))
        ax1.set_title('Original Image', fontsize=25)
        ax2.imshow(img)
        ax2.set_title('With Corners', fontsize=25)
        #plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()

#cv2.destroyAllWindows()
