'''
Given a video, output the labels file on x-y coordinate
'''
import os
import time
import numpy as np
import cv2
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh\

from bgrwarp import mat_from_img
target_size = (656, 368)
if __name__ == '__main__':
    src = 'videos/vid1.webm'
    video = cv2.VideoCapture(src)
    fps = video.get(cv2.CAP_PROP_FPS)

    # Use the first frame to get matrix transform and dimensions
    ret_val, frame1 = video.read()
    frame1 = cv2.resize(frame1, target_size)
    M, (width, height) = mat_from_img(frame1)

