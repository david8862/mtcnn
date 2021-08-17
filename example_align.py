#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# MIT License
#
# Copyright (c) 2019 Iván de Paz Centeno
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import cv2
#from skimage import transform as trans
from mtcnn import MTCNN

# reference from
# https://github.com/deepinsight/insightface/blob/f61956fda322db9977a9bc250031c079c2eb6192/src/common/face_preprocess.py
def align_face(image, bbox, keypoints, face_size=(112, 96), show_point=True):
    # check face size
    # only support (112,96) and (112,112)
    assert len(face_size) == 2
    assert face_size[0] == 112 and (face_size[1] == 112 or face_size[1] == 96)

    # target 5 (x,y) format keypoints in (112,96) shape,
    # keypoint order: (left_eye, right_eye, nose, mouth_left, mouth_right)
    target_points = np.array([[30.2946, 51.6963],
                              [65.5318, 51.5014],
                              [48.0252, 71.7366],
                              [33.5493, 92.3655],
                              [62.7299, 92.2041]], dtype=np.float32)
    # if use (112, 112) shape, adjust x coordinate
    if face_size[1] == 112:
        target_points[:,0] += 8.0

    # src keypoints array
    src_points = keypoints.astype(np.float32)

    # calculate transform matrix
    M, _ = cv2.estimateAffine2D(src_points.reshape(1,5,2), target_points.reshape(1,5,2))
    #tform = trans.SimilarityTransform()
    #tform.estimate(src_points, target_points)
    #M = tform.params[0:2,:]

    # do affine transformation to get aligned face image
    aligned_face = cv2.warpAffine(image, M, (face_size[1], face_size[0]), borderValue=0.0)

    # show keypoints on face image
    if show_point:
        for target_point in target_points:
            cv2.circle(aligned_face, tuple(target_point), 1, (0, 155, 255), 2)

    return aligned_face



detector = MTCNN()

cv2.namedWindow('MTCNN', cv2.WINDOW_NORMAL)
cv2.namedWindow('Aligned Face', cv2.WINDOW_NORMAL)

image = cv2.cvtColor(cv2.imread("ivan.jpg"), cv2.COLOR_BGR2RGB)
results = detector.detect_faces(image)

# only align 1st face
result = results[0]
bounding_box = result['box']
keypoints = result['keypoints']

# bbox in (xmin,ymin,xmax,ymax) format
bbox = (bounding_box[0], bounding_box[1], bounding_box[0]+bounding_box[2], bounding_box[1]+bounding_box[3])
# MTCNN 5 keypoint array
keypoints_array = np.asarray([keypoints['left_eye'], keypoints['right_eye'], keypoints['nose'], keypoints['mouth_left'], keypoints['mouth_right']])

face = align_face(image, bbox=bbox, keypoints=keypoints_array, face_size=(112, 96))

cv2.imshow("Aligned Face", cv2.cvtColor(face, cv2.COLOR_RGB2BGR))



for result in results:
    # Result is an array with all the bounding boxes detected. We know that for 'ivan.jpg' there is only one.
    bounding_box = result['box']
    keypoints = result['keypoints']

    cv2.rectangle(image,
                  (bounding_box[0], bounding_box[1]),
                  (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                  (0,155,255),
                  2)

    cv2.circle(image,(keypoints['left_eye']), 2, (0,155,255), 2)
    cv2.circle(image,(keypoints['right_eye']), 2, (0,155,255), 2)
    cv2.circle(image,(keypoints['nose']), 2, (0,155,255), 2)
    cv2.circle(image,(keypoints['mouth_left']), 2, (0,155,255), 2)
    cv2.circle(image,(keypoints['mouth_right']), 2, (0,155,255), 2)

cv2.imshow("MTCNN", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
#cv2.imwrite("timg_drawn.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

print(results)
