# https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
import numpy as np
import cv2
import time
import argparse
from tf_pose.networks import model_wh, get_graph_path
from tf_pose.estimator import TfPoseEstimator

pts_array = []
selected = 0
coord = None
warped_dimensions = None
room_size = None
warp_click = None
def _order_points(pts):
    """
    0       1

    3       2
    """
    pts = np.array(pts, dtype='float32')
    rect = np.zeros(shape=(4, 2), dtype='float32')
    # we now find point 0 and point 2, which we just do by summing elements axis = 1
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now to find 1 and 3
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def get_points(event, x, y, flags, param):
    'Select 4 points on the image to get the perspective transform'
    global pts_array, selected
    if event == cv2.EVENT_LBUTTONDOWN:
        pts_array.append([x, y])
        selected += 1


def get_coordinates(event, x, y, flags, param):
    'Get the actual measurement of any point in the image given the dimensions of the room'
    global coord, warp_click
    # opencv height width
    # our implementation width height

    if event == cv2.EVENT_LBUTTONDOWN:
        realx = room_size[0] * x / warped_dimensions[1]
        realy = room_size[1] * y / warped_dimensions[0]
        coord = (realx, realy)
        warp_click = (x, y)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resize', type=str, default='1280x720')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--model', type=str, default='mobilenet_thin')
    parser.add_argument('--roomsize', type=str, default='5x5')
    args = parser.parse_args()

    w, h = model_wh(args.resize)
    rw, rh = args.roomsize.split('x')
    room_size = (float(rw), float(rh))
    print(room_size)
    #e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    cam = cv2.VideoCapture(args.camera)

    # Here, we setup the 4 corner points of the room
    print('SPACE to capture...')
    while True:
        _, image = cam.read()
        cv2.imshow('test', image)

        key = cv2.waitKey(1)
        if key % 256 == 32:
            # Space
            cv2.destroyAllWindows()
            break

        if key % 256 == 27:
            # ESC
            exit(1)

    _, origin = cam.read()
    clone = origin.copy()
    cv2.namedWindow('origin')
    cv2.setMouseCallback('origin', get_points)

    while selected < 4:
        cv2.imshow('origin', clone)
        key = cv2.waitKey(1)
        for pts in pts_array:
            cv2.circle(clone, (pts[0], pts[1]), 3, (0, 0, 255), -1)

        if key == ord('r'):
            pts_array = []
            selected = 0
            clone = origin.copy()

    print(pts_array)
    cv2.destroyAllWindows()
    fps_time = 0
    cv2.namedWindow('warped')
    cv2.setMouseCallback('warped', get_coordinates)
    while True:
        ret_val, image = cam.read()
        cv2.putText(image,
                    "FPS: {:.2f}".format((1.0 / (time.time() - fps_time))),
                    (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        for pts in pts_array:
            cv2.circle(image, (pts[0], pts[1]), 3, (0, 0, 255), -1)
        cv2.imshow('original', image)
        warped = four_point_transform(image, pts_array)
        warped_dimensions = warped.shape[:2]
        if coord is not None:
            cv2.putText(warped, 'x: {:.2f}m,  y: {:.2f}m'.format(coord[0], coord[1]),
                        (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)
            cv2.circle(warped, (warp_click[0], warp_click[1]), 3, (0, 0, 255), -1)
        cv2.imshow('warped', warped)

        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            # ESC
            break
    cv2.destroyAllWindows()