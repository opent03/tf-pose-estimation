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
from pre_label import interpolate
target_size = (656, 368)

def warp(mat, point, dims):
    point = np.array([[point[0], point[1]]])
    point = point.reshape((-1,1,2))
    newpoint = cv2.perspectiveTransform(point, mat)
    newpoint = newpoint.reshape(2,)
    return newpoint

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='video_labeling')
    parser.add_argument('--video', type=str, default='viddat/pos10/pos10circ.mov')
    parser.add_argument('--model', type=str, default='mobilenet_thin',
                        help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--warped_labels', type=bool, default=True,
                        help='labels to be in warped coordinates or original coordinates?')
    parser.add_argument('--out', type=str, default='unnamed_label.dat',
                        help='name of the label file')
    parser.add_argument('--flipvideo', type=bool, default=False, help='Does exactly what you think it does')
    parser.add_argument('--labels_dir', type=str, default='labels', help='labels directory')
    args = parser.parse_args()

    labels_dir = args.labels_dir
    video = cv2.VideoCapture(args.video)
    if video.isOpened() is False:
        print("Error opening video stream or file")
        exit(1)

    # METADATA
    fps = video.get(cv2.CAP_PROP_FPS)
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print('duration: {}'.format(round(length/fps, 0)))

    # Get matrix transform and dimensions
    ret_val, frame1 = video.read()
    frame1 = cv2.resize(frame1, target_size)
    print(args.flipvideo)
    if args.flipvideo:
        frame1 = cv2.flip(frame1, -1)
    M, dims = mat_from_img(frame1)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=target_size)
    framecount = 0

    with open(os.path.join(labels_dir, args.out), 'w+') as f:
        counter = 0
        label_rate = 0.2 # interval in seconds between each label
        f.write('{}\t{}\t{}\n'.format(fps, label_rate, args.video))
        while True:
            ret_val, frame = video.read()
            framecount += 1
            if not ret_val:
                break
            frame = cv2.resize(frame, target_size)
            if args.flipvideo:
                frame = cv2.flip(frame, -1)
            counter += 1
            if counter >= fps*label_rate:
                counter = 0
                humans = e.inference(frame)
                try:
                    loc_list = TfPoseEstimator.get_xy(human=humans[0])
                    midpt = TfPoseEstimator.get_midpt(loc_list[0], loc_list[1])
                    if args.warped_labels:
                        midpt = (midpt[0]*target_size[0], midpt[1]*target_size[1])
                        midpt = warp(M, midpt, dims)
                        midpt = midpt[0]/dims[0], midpt[1]/dims[1]
                    f.write('({},{})\n'.format(midpt[0], midpt[1]))
                except Exception as E:
                    midipt = (0,0)
                    f.write('NaN\n')

                if round((framecount/length)*100, 0) % 10 == 0:
                    print('{}%'.format(round((framecount/length)*100, 0)))
                    frame = TfPoseEstimator.draw_humans(frame, humans)

                    print('midpoint: ({},{})'.format(midpt[0], midpt[1]))
                    '''
                    cv2.imshow('actual', frame)
                    warped = cv2.warpPerspective(frame, M, dims)
                    cv2.imshow('warped', warped)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    '''