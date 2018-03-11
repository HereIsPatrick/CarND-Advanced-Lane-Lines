import cv2
import numpy as np
from libs import camera_calibration
from libs.fit_line import *
from libs.threshold import *

class Pipeline:
    line = None
    m = None
    minv = None
    mat = None
    dist = None

    @staticmethod
    def set_values(l, m, minv, mat, dist):
        Pipeline.line = l
        Pipeline.m = m
        Pipeline.minv = minv
        Pipeline.mat = mat
        Pipeline.dist = dist

    @staticmethod
    def pipeline(img):
        line, m, minv, mat, dist = Pipeline.line, Pipeline.m, Pipeline.minv, Pipeline.mat, Pipeline.dist
        if (line is None or m is None or minv is None or mat is None or dist is None):
            raise NotImplementedError
            
        img_size = (img.shape[1], img.shape[0])
        width, height = img_size


        # Step. undistort image
        img = camera_calibration.undistort(np.copy(img), mat, dist)

        # Step. warp
        binary_warped = cv2.warpPerspective(combine_threshold(img), m, (width, height))
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        
        nwindows = 9
        window_height = np.int(binary_warped.shape[0]/nwindows)
        margin = 100
        minpix = 50
        
        if not line.is_first_processed:
            histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)

            midpoint = np.int(histogram.shape[0]/2)
            leftx_base = np.argmax(histogram[:midpoint])
            rightx_base = np.argmax(histogram[midpoint:]) + midpoint

            nonzero = binary_warped.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])

            leftx_current = leftx_base
            rightx_current = rightx_base

            # right lane pixel indices
            left_lane_inds = []
            right_lane_inds = []

            # Step. through the windows one by one
            for window in range(nwindows):
                win_y_low = binary_warped.shape[0] - (window+1)*window_height
                win_y_high = binary_warped.shape[0] - window*window_height
                win_xleft_low = leftx_current - margin
                win_xleft_high = leftx_current + margin
                win_xright_low = rightx_current - margin
                win_xright_high = rightx_current + margin

                accept_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high)
                                    & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
                accept_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high)
                                    & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

                left_lane_inds.append(accept_left_inds)
                right_lane_inds.append(accept_right_inds)

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
            line.update_fit(left_fit, right_fit)

            line.is_first_processed = True
     
        else:
            nonzero = binary_warped.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])

            left_fit = line.left_fit
            right_fit = line.right_fit
            left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
            right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

            # Step. get right & left pixels in the window.
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds] 
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]
            
            # Step . ceate the second order polynomial for left & right.
            line.update_fit(np.polyfit(lefty, leftx, 2), np.polyfit(righty, rightx, 2))
            left_fit = line.left_fit
            right_fit = line.right_fit
           
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Step . transfer to fillpoly format.
        left_pts = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        right_pts = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((left_pts, right_pts))

        # Step . draw lane & fill color green
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        new_warp = cv2.warpPerspective(color_warp, minv, (img.shape[1], img.shape[0]))

        # Step. combine with original image
        result = cv2.addWeighted(img, 1, new_warp, 0.3, 0)
        
        # Step.  put radius of curvature info
        cv2.putText(result,'Radius of Curvature: %.2fm' % line.curvature,(20,40), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
        
        # Step . Add distance from center
        position_from_center = line.get_distance_from_center()
        if position_from_center < 0:
            text = 'left'
        else:
            text = 'right'
        cv2.putText(result,'%.2fm %s of center' % (np.absolute(position_from_center), text),(20,80), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
        
        return result
