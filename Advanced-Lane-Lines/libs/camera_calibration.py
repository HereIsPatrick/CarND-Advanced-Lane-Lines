import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math


def get_calibration_parameter(path):
    object_points = []  #3d
    img_points = []  #2d

    # Step . Get all images.
    images = glob.glob(path)
    total_image_count = len(images)

    image_count = 1
    fig = plt.figure()
    for filename in images:
        img = cv2.imread(filename)
        nx, ny = 6, 9
        retval, corners = cv2.findChessboardCorners(img, (nx, ny))
        objp = np.zeros((nx * ny, 3), np.float32)
        objp[:, :2] = np.mgrid[0: nx, 0: ny].T.reshape(-1, 2)

        # Step . collect the image that with 9x6 chessboard.
        if retval:
            object_points.append(objp)
            img_points.append(corners)

            chessboard_with_corners = cv2.drawChessboardCorners(img, (nx, ny), corners, retval)
            chessboard_with_corners = cv2.cvtColor(chessboard_with_corners, cv2.COLOR_BGR2RGB)

            image_count += 1

    return cv2.calibrateCamera(object_points, img_points, img.shape[0:2], None, None), fig


def undistort(img, mat, dist):
    return cv2.undistort(img, mat, dist)

