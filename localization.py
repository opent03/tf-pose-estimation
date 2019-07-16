import sys
import os
import time
from tf_pose import common
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import matplotlib.pyplot as plt

from bgrwarp import four_point_transform


def get_points(event, x, y, flags, param):
    'Select 4 points on the image to get the perspective transform'
    global pts_array, selected
    if event == cv2.EVENT_LBUTTONDOWN:
        pts_array.append([x, y])
        selected += 1


# same coordinates
pts_array = []
selected = 0

imgpath = 'photos'
# Width, height
target_size = (656, 368)
images = [os.path.join(imgpath, i) for i in ['photo{}.jpg'.format(k) for k in range(1, 7)]]
images = [cv2.resize(i, target_size) for i in [cv2.imread(j) for j in images]]
original = images[0]



real_dims = (2.0, 2.0)
cv2.namedWindow('origin')
cv2.setMouseCallback('origin', get_points)
while selected < 4:
    cv2.imshow('origin', original)
    k = cv2.waitKey(1)
    for pts in pts_array:
        cv2.circle(original, (pts[0], pts[1]), 3, (0, 0, 255), -1)

print(pts_array)
warped, tp = four_point_transform(original, pts_array)
transform_matrix, dims = tp
cv2.destroyAllWindows()

warped = [cv2.warpPerspective(i, transform_matrix, dims)
          for i in images]
'''
cv2.imshow('new', warped[5])
cv2.imshow('lmao', images[5])
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
# inference
model_path = 'mobilenet_thin'
e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=target_size)
t = time.time()
humans = e.inference(images[1], upsample_size=4.0)
viet = humans[0]
elapsed = time.time() - t
displayed = TfPoseEstimator.draw_humans(images[1], humans, imgcopy=False)
try:
    loc_list = TfPoseEstimator.get_xy(viet)
    loc_list = list(map(lambda x: (x[0]*target_size[0], x[1]*target_size[1]), loc_list))
    print(loc_list)
    midpt = TfPoseEstimator.get_midpt(loc_list[0], loc_list[1])
    midpt = list(map(lambda x: int(x), (midpt[0], midpt[1])))
    print(midpt)
    cv2.circle(displayed, tuple(midpt), 3, (255, 123, 85), thickness=3)
except:
    pass

cv2.namedWindow('humans')
cv2.imshow('humans', displayed)
cv2.namedWindow('humans_warped')
humans_warped = cv2.warpPerspective(images[1], transform_matrix, dims)
cv2.imshow('humans_warped', humans_warped)
cv2.waitKey(0)
cv2.destroyAllWindows()