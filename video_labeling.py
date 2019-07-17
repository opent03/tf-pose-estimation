'''
Given a video, output the labels file on x-y coordinate
'''
import os
import time
import numpy as np
import argparse
import cv2
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path

from bgrwarp import mat_from_img
target_size = (656, 368)

def warp(mat, point, dims):
    point = np.array([[point[0], point[1]]])
    point = point.reshape((-1,1,2))
    newpoint = cv2.perspectiveTransform(point, mat)
    newpoint = newpoint.reshape(2,)
    return newpoint

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='video_labeling')
    parser.add_argument('--video', type=str, default='videos/vid1.mov')
    parser.add_argument('--model', type=str, default='mobilenet_thin',
                        help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--warped_labels', type=bool, default=True,
                        help='labels to be in warped coordinates or original coordinates?')
    args = parser.parse_args()


    labels_dir = 'labels'
    video = cv2.VideoCapture(args.video)
    fps = video.get(cv2.CAP_PROP_FPS)

    # Use the first frame to get matrix transform and dimensions
    ret_val, frame1 = video.read()
    frame1 = cv2.resize(frame1, target_size)
    M, (width, height) = mat_from_img(frame1)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=target_size)

    with open(os.path.join(labels_dir, 'label1.dat'), 'w+') as f:
        counter = 0
        label_rate = 0.2 # interval in seconds between each label
        f.write('{}\t{}\t{}\n'.format(fps, label_rate, args.video))

        while True:
            ret_val, frame = video.read()
            if not ret_val:
                break
            frame = cv2.resize(frame, target_size)
            counter += 1
            if counter >= fps*label_rate:
                counter = 0
                humans = e.inference(frame)
                try:
                    loc_list = TfPoseEstimator.get_xy(human=humans[0])
                    midpt = TfPoseEstimator.get_midpt(loc_list[0], loc_list[1])
                    if args.warped_labels:
                        midpt = (midpt[0]*target_size[0], midpt[1]*target_size[1])
                        midpt = warp(M, midpt, (width, height))
                        midpt = midpt[0]/target_size[0], midpt[1]/target_size[1]
                        print(midpt)
                    f.write('({},{})\n'.format(midpt[0], midpt[1]))
                    print('wrote successfully')
                except Exception as E:
                    print(E)
                    f.write('NaN\n')

