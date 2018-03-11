import numpy as np
import cv2

def direction_threshold(img, sobel_kernel=15, thresh=(0.7, 1.3)):
    # Step. get gray image
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    # Step. get gradients of sobel x,y
    sobelx = cv2.Sobel(gray[:,:,2], cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray[:,:,2], cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # Step. use arctan to get direction of gradient.
    grad = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary =  np.zeros_like(grad)
    binary[(grad >= thresh[0]) & (grad <= thresh[1])] = 1

    return binary


def magnitude_threshold(img, sobel_kernel=9, mag_thresh=(30, 255)):
    # Step. get gray image
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    # Step. get gradients of sobel x,y
    sobelx = cv2.Sobel(gray[:,:,2], cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray[:,:,2], cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # Step. calculate the magnitude gradient
    gradmag = np.sqrt(sobelx**2 + sobely**2)

    # Step. scale to 8bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Step. create image, gradmag between 30 to 255, fill 1
    binary = np.zeros_like(gradmag)
    binary[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    return binary


def combine_threshold(img, paint_color=False, mag_dir_thresh=False):

    img = np.copy(img)
    
    # Step . convert to HLS space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    
    # Step . create white and yellow mask
    ## White
    lower_white = np.array([0,220,0], dtype=np.uint8)
    upper_white = np.array([255,255,255], dtype=np.uint8)
    white_mask = cv2.inRange(hls, lower_white, upper_white)
    
    ## Yellow
    lower_yellow = np.array([20,0,100], dtype=np.uint8)
    upper_yellow = np.array([32,220,255], dtype=np.uint8)
    yellow_mask = cv2.inRange(hls, lower_yellow, upper_yellow)  
    
    combined_binary = np.zeros_like(white_mask)
    
    # Step. combine direction & magnitude mask.
    if mag_dir_thresh:
        dir_mask = direction_threshold(img)
        mag_mask = magnitude_threshold(img)
        combined_binary[((dir_mask == 1) & (mag_mask == 1))] = 255
        
    if paint_color:
        return np.dstack((white_mask, yellow_mask, combined_binary))
    else:
        combined_binary[((white_mask == 255) | (yellow_mask == 255))] = 255
        combined_binary[(combined_binary == 255)] = 1
        return combined_binary