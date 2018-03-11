import cv2
import numpy as np
import pickle
import glob
import matplotlib.pyplot as plt

objp = np.zeros((6 * 9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# 3d and 2d array
objpoints = []  #3d
imgpoints = []  #2d


images = glob.glob('../camera_cal/calibration*.jpg')

fig, axs = plt.subplots(5, 4, figsize=(16, 13.5))

axs = axs.ravel()

err_count = 0
for i, file_name in enumerate(images):

    img = cv2.imread(file_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Step. Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

    # Step. check
    if ret == True:
        # Step . found
        objpoints.append(objp)
        imgpoints.append(corners)

        # Step. display the corners
        img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
        axs[i].imshow(img)
        axs[i].set_title(file_name)
    else:
        # Step. not found.
        err_count+=1
        axs[i].set_title(file_name)

print('We have {} images that corner is not 9x6'.format(err_count))
plt.show()

#---------------------------------------------------------------------------
# undistored a image
# Step. read test image 1
img = cv2.imread('../camera_cal/calibration1.jpg')
img_size = (img.shape[1], img.shape[0])

# Step. calibrate camera.
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

# Step. undistorted test image.
dst = cv2.undistort(img, mtx, dist, None, mtx)

# Step. Save the camera calibration parameter
cal_parameter_pickle = {}
cal_parameter_pickle["mtx"] = mtx
cal_parameter_pickle["dist"] = dist
pickle.dump( cal_parameter_pickle, open( "calibration.p", "wb" ) )

# Step. display
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(18,9))

ax1.imshow(img)
ax1.set_title('Original', fontsize=32)
ax2.imshow(dst)
ax2.set_title('Undistorted', fontsize=32)
plt.show()

#---------------------------------------------------------------------------

# Step. read test image 1
#test_img = cv2.imread('../test_images/straight_lines2.jpg')
test_img = cv2.imread('../test_images/test6.jpg')

test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

# Step. undistort image
undist_img = cv2.undistort(test_img, mtx, dist, None, mtx)

# Step. display
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(test_img)
ax1.set_title('Original', fontsize=30)
ax2.imshow(undist_img)
ax2.set_title('Undistorted', fontsize=30)

plt.show()

#---------------------------------------------------------------------------
from libs.threshold import *

# Step . generate direction & magnitude mask
plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
thresholded = combine_threshold(undist_img, paint_color=True)
plt.imshow(thresholded)
plt.title('Different Mask color(Magnitude & direction)')

# Step . combine two mask into one mask.
plt.subplot(1,2,2)
thresholded_binary_img = combine_threshold(undist_img)
plt.imshow(thresholded_binary_img, cmap='gray')
plt.title('Thresholded Binary')

plt.show()

#---------------------------------------------------------------------------
img_size = (thresholded_binary_img.shape[1], thresholded_binary_img.shape[0])
width, height = img_size
offset = 200
src = np.float32([
    [  563,   455 ],
    [  720,   455 ],
    [ 1130,   720 ],
    [  190 ,   720 ]])
dst = np.float32([[offset, 0], [img_size[0] - offset, 0], [img_size[0] - offset, img_size[1]], [offset, img_size[1]]])
m = cv2.getPerspectiveTransform(src, dst)
minv = cv2.getPerspectiveTransform(dst, src)

plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
thresholded_binary_img = combine_threshold(undist_img)
a=src

# Step. plot source points, for perspective transform.
plt.plot(*zip(*a), marker='o', color='r', ls='')
plt.imshow(thresholded_binary_img, cmap='gray')
plt.title('Thresholded Binary')

plt.subplot(1,2,2)
binary_warped_img = cv2.warpPerspective(thresholded_binary_img, m, (width, height))
plt.imshow(binary_warped_img, cmap='gray')
plt.title('Binary Warped Image')

plt.show()
#---------------------------------------------------------------------------
# Step . histogram  half of the image
histogram = np.sum(binary_warped_img[binary_warped_img.shape[0] // 2:, :], axis=0)

# Step . create output image
out_img = np.dstack((binary_warped_img, binary_warped_img, binary_warped_img)) * 255

# Step. finding peak on the histogram, can find the start points of left and right lines
midpoint = np.int(histogram.shape[0] / 2)
leftx_base = np.argmax(histogram[:midpoint])
rightx_base = np.argmax(histogram[midpoint:]) + midpoint

# slide windows
nwindows = 9

# Height of windows
window_height = np.int(binary_warped_img.shape[0] / nwindows)

# Step. check nonzero pixels in the image(for x, y)
nonzero = binary_warped_img.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])

# Step. current positions for each window
leftx_current = leftx_base
rightx_current = rightx_base

# margin for search range
margin = 100

# mininal pixels found that recorrect to center window
minpix = 50

# Empty lists to receive left and right lane pixel indices
left_lane_inds = []
right_lane_inds = []

# Step. check windows one by one
for window in range(nwindows):
    # Step. identify window boundaries
    win_y_low = binary_warped_img.shape[0] - (window + 1) * window_height
    win_y_high = binary_warped_img.shape[0] - window * window_height
    win_xleft_low = leftx_current - margin
    win_xleft_high = leftx_current + margin
    win_xright_low = rightx_current - margin
    win_xright_high = rightx_current + margin

    # Step. draw window
    cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
    cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

    # Step. check nonzero pixels in the image(for x, y)
    accept_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
                nonzerox < win_xleft_high)).nonzero()[0]
    accept_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                nonzerox < win_xright_high)).nonzero()[0]

    # Step. appending nonzero pixels
    left_lane_inds.append(accept_left_inds)
    right_lane_inds.append(accept_right_inds)

    # Step. recorrect next center position
    if len(accept_left_inds) > minpix:
        leftx_current = np.int(np.mean(nonzerox[accept_left_inds]))
    if len(accept_right_inds) > minpix:
        rightx_current = np.int(np.mean(nonzerox[accept_right_inds]))

# Step. Concatenate the arrays
left_lane_inds = np.concatenate(left_lane_inds)
right_lane_inds = np.concatenate(right_lane_inds)

# Step. get right & left pixels in the window.
leftx = nonzerox[left_lane_inds]
lefty = nonzeroy[left_lane_inds]
rightx = nonzerox[right_lane_inds]
righty = nonzeroy[right_lane_inds]

# Step . ceate the second order polynomial for left & right.
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)

## Step. display
ploty = np.linspace(0, binary_warped_img.shape[0] - 1, binary_warped_img.shape[0])
left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

# Step. paint out image with lef & right side by different color pixel
out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.imshow(binary_warped_img, cmap='gray')
plt.title('Binary Warped')

plt.subplot(1, 2, 2)
plt.imshow(out_img)
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')
plt.xlim(0, 1280)
plt.ylim(720, 0)
plt.title('Lane Detected')
plt.show()

#---------------------------------------------------------------------------

ym_per_pix = 3.0/72.0 # meter / pixel(y)
xm_per_pix = 3.7/700.0 # meter / pixel(x)
y_eval = 700
midx = 650

y1 = (2*left_fit[0]*y_eval + left_fit[1])*xm_per_pix/ym_per_pix
y2 = 2*left_fit[0]*xm_per_pix/(ym_per_pix*ym_per_pix)

#Radius of Curvature and Distance from Lane Center Calculation" using this line of code (altered for clarity):
curvature = ((1 + y1*y1)**(1.5))/np.absolute(y2)
print("Radius of Curvature: %f" % curvature)

#---------------------------------------------------------------------------

ploty = np.linspace(0, binary_warped_img.shape[0] - 1, binary_warped_img.shape[0])
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

warp_zero = np.zeros_like(binary_warped_img).astype(np.uint8)
color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

# Step . transfer to fillpoly format.
left_pts = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
right_pts = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
pts = np.hstack((left_pts, right_pts))

# Step . draw lane & fill color green
cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
new_warp = cv2.warpPerspective(color_warp, minv, (undist_img.shape[1], undist_img.shape[0]))

# Step. combine with original image
result = cv2.addWeighted(undist_img, 1, new_warp, 0.3, 0)

# Step.  put radius of curvature info
cv2.putText(result,'Radius of Curvature: %.2fm' % curvature,(20,40), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)

# Step . calculate distance from center
x_left_pix = left_fit[0]*(y_eval**2) + left_fit[1]*y_eval + left_fit[2]
x_right_pix = right_fit[0]*(y_eval**2) + right_fit[1]*y_eval + right_fit[2]
position_from_center = ((x_left_pix + x_right_pix)/2 - midx) * xm_per_pix
if position_from_center < 0:
    text = 'left'
else:
    text = 'right'
cv2.putText(result,'%.2fm %s of center' % (np.absolute(position_from_center), text),(20,80), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
plt.imshow(result)
plt.show()

#---------------------------------------------------------------------------
from libs.pipeline import Pipeline
from libs.fit_line import FitLine
from moviepy.editor import VideoFileClip

# Step. create fit line.
fit_line=FitLine()

# Step. set parameter.
Pipeline.set_values(fit_line, m, minv, mtx, dist)

vfc = VideoFileClip("../project_video.mp4")

# Step. fill image to pipeline to process it.
white_clip = vfc.fl_image(Pipeline.pipeline)

# Step. save result to video file.
white_clip.write_videofile('project_video_output.mp4', audio=False)

