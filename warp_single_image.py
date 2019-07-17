import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path

from bgrwarp import mat_from_img
image = cv2.imread('../Desktop/img1.jpg')
print(image.shape)
M, dims = mat_from_img(image)
cv2.namedWindow('original')
cv2.namedWindow('warped')
cv2.imshow('original', image)
warped = cv2.warpPerspective(image, M, dims)
cv2.imshow('warped', warped)
cv2.waitKey(0)
cv2.destroyAllWindows()