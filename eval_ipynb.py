# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import tensorflow as tf

import time

import itertools

import cPickle

import os

import matplotlib.pyplot as plt
plt.subplots_adjust(wspace=0.01, hspace=0.01,
                    left=0.01, right=0.99,
                    bottom=0.01, top=0.99)

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import skimage.transform as trans
from skimage.feature import peak_local_max
from skimage.morphology import dilation

import matplotlib.patches as patches

import nn
import models
import utils


def resize(img):
  return (trans.resize(img, [224, 224]) * 255).astype('uint8')

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def IoU(box1, box2):

  def inter(x1, x2, y1, y2):
    if y1 < x1:
      x1, x2, y1, y2 = y1, y2, x1, x2
    if y1 > x2:
      return 0
    return min(x2, y2) - y1

  def area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


  inter_area = (inter(box1[0], box1[2], box2[0], box2[2]) *
                inter(box1[1], box1[3], box2[1], box2[3]))


  return inter_area / float((area(box1) + area(box2) - inter_area))

def get_vg_boxes(a, H, W):
  a = np.array(a).reshape([2, 4])
  objects = a[0, :].astype('float32')
  subjects = a[1, :].astype('float32')

  subjects[[2, 3]] += subjects[[0, 1]]
  subjects[[0, 2]] /= H
  subjects[[1, 3]] /= W

  objects[[2, 3]] += objects[[0, 1]]
  objects[[0, 2]] /= H
  objects[[1, 3]] /= W

  return subjects, objects

def get_pred_boxes(a):
  subjects = [np.where(a[0, :, :, 0])[0][0] / 54.0,
              np.where(a[0, :, :, 0])[1][0] / 54.0,
              np.where(a[0, :, :, 1])[0][0] / 54.0,
              np.where(a[0, :, :, 1])[1][0] / 54.0]
  objects = [np.where(a[0, :, :, 2])[0][0] / 54.0,
             np.where(a[0, :, :, 2])[1][0] / 54.0,
             np.where(a[0, :, :, 3])[0][0] / 54.0,
             np.where(a[0, :, :, 3])[1][0] / 54.0]
  return subjects, objects

def fix_box(b):
  if b[0] > b[2]:
    b[0], b[2] = b[2], b[0]
  if b[1] > b[3]:
    b[1], b[3] = b[3], b[1]
  return np.array(b)

def dil(x):
  return dilation(dilation(x))


def model_template(images, labels,
                   boxes,
                   stage):
  return models.model_detection(images, labels,
                                boxes,
                                stage)

model_factory = tf.make_template("detection", model_template)

imgs_ph = tf.placeholder(shape=[None, 224, 224, 3], dtype=tf.float32)
class_ph = tf.placeholder(shape=[None, 41], dtype=tf.float32)
boxes_ph = tf.placeholder(shape=[None, 56, 56, 4], dtype=tf.float32)
stage_ph = tf.placeholder(shape=[], dtype=tf.int32)

tf.GLOBAL = {}
tf.GLOBAL["init"] = True
tf.GLOBAL["dropout"] = 0.0

with tf.device("/cpu:0"):
  _ = model_factory(imgs_ph, [class_ph], boxes_ph, stage_ph)

tf.GLOBAL["init"] = False
tf.GLOBAL["dropout"] = 0.0

with tf.device("gpu:0"):
      [label_p_v, point_p_v], loss = model_factory(imgs_ph, [class_ph],
                                                   boxes_ph, stage_ph)

dataset = cPickle.load(open('/VG/pickle/val.pickle'))
prefix = '/VG/images/'

saver = tf.train.Saver()

box_count = 0
pos = 0
guesses = 0
g_list = []

vis = 1

try:
  os.mkdir('/tmp/results/correct/')
except:
  pass

f1 = plt.figure(figsize=(25, 10))
f2 = plt.figure(figsize=(10, 5))


with tf.Session() as sess:
  # Restore the trained model
  saver.restore(sess, '')

  # Loop over validation examples
  for ex_count, ex in enumerate(dataset):
    # Unpack example
    filename, labels_gt, boxes_gt = ex

    # Read image, check shape, get shape
    im = pylab.imread(os.path.join(prefix, filename))
    if len(im.shape) != 3:
      continue
    H, W = im.shape[:2]

    # Preprocess GT boxes
    boxes_gt = [get_vg_boxes(b, H, W) for b in
                np.split(np.array(boxes_gt), len(boxes_gt) / 8)]

    # Preprocess input image
    im = (resize(im)[None] - 127.5) / 127.5

    # stage 0
    # p = sess.run(label_p_v, {imgs_ph: im,
    #                          class_ph: label_np,
    #                          boxes_ph: boxes_np,
    #                          stage_ph: 0})[0]
    # p = softmax(p)
    # labels_pred = set(list(np.where(p[0] > (0.2 * np.max(p[0])))[0]))

    for label in set(labels_gt):

      label_np = np.zeros((1, 41))
      label_np[0, label] = 1

      def explore(box_np, corner_np, stage):

        l = sess.run(point_p_v, {imgs_ph: im,
                                 class_ph: label_np,
                                 boxes_ph: box_np,
                                 stage_ph: stage + 1})[stage]
        corner_np[0, :, :, stage] = softmax(l, axis=(1, 2))[0, :, :, 0]

        peaks = peak_local_max(softmax(l, axis=(1, 2))[0, :, :, 0],
                               min_distance=1,
                               threshold_rel=0.1, exclude_border=False,
                               num_peaks=4 - 2 * stage % 2)

        results = []
        for peak in peaks:
          box_np[:, peak[0], peak[1], stage] = 1

          if stage == 3:
            results.append((np.array(box_np), np.array(corner_np)))
            box_np[:, peak[0], peak[1], stage] = 0
          else:
            results += explore(np.array(box_np), np.array(corner_np),
                               stage + 1)
            box_np[:, peak[0], peak[1], stage] = 0

        return results

      box_np = np.zeros((1, 56, 56, 4))
      corner_np = np.zeros((1, 56, 56, 4))

      result = explore(box_np, corner_np, 0)
      boxes_pred = [x[0] for x in result]
      corners_pred = [x[1] for x in result]

      # Visualize predictions

      if (200 <= ex_count <= 400):

        # Create image dir
        try:
          os.mkdir('/tmp/results/%03d' % ex_count)
        except:
          pass

        for ii, (box, soft_corner) in enumerate(zip(boxes_pred, corners_pred)):

          # Get predicted boxes
          [subject_pred, object_pred] = [np.array(fix_box(b))
                                         for b in get_pred_boxes(box)]

          # show original image
          ax = f1.add_subplot(2, 5, 1)
          ax.grid('off')
          ax.axis('off')
          ax.imshow((im[0] * 127.5 + 127.5).astype('uint8'))

          # show relationship detection
          ax = f1.add_subplot(2, 5, 6)
          ax.grid('off')
          ax.axis('off')
          ax.imshow((im[0] * 127.5 + 127.5).astype('uint8'))

          ax.add_patch(patches.Rectangle(object_pred[0:2][::-1] * 224,
                                         (object_pred[3] - object_pred[1]) * 224,
                                         (object_pred[2] - object_pred[0]) * 224,
                                         fill=False,
                                         linewidth=7, color='red'))

          ax.add_patch(patches.Rectangle(subject_pred[0:2][::-1] * 224,
                                         (subject_pred[3] - subject_pred[1]) * 224,
                                         (subject_pred[2] - subject_pred[0]) * 224,
                                          fill=False,
                                          linewidth=4, color='blue'))

          for i in range(4):
            ax = f1.add_subplot(2, 5, 7 + i)
            ax.grid('off')
            ax.axis('off')
            ax.matshow(dil(box[0, :, :, i]), cmap=plt.get_cmap('jet'))

          for i in range(4):
            ax = f1.add_subplot(2, 5, 2 + i)
            ax.grid('off')
            ax.axis('off')
            ax.set_title('Stage %i' % (i + 1))
            ax.matshow(soft_corner[0, :, :, i], cmap=plt.get_cmap('jet'))

          f1.subplots_adjust(wspace=0.01, hspace=0.01,
                            left=0.05, right=0.95,
                            bottom=0.05, top=0.95)

          f1.savefig('/tmp/results/%03d/%s-%03d.jpg' % (ex_count,
                                                       utils.LABEL_MAP[label], ii))

          f1.clf()

      correct_detections = np.zeros(len(labels_gt))

      guesses += len(boxes_pred)
      g_list.append(len(boxes_pred))

      # For every predicted box
      for box in boxes_pred:

        # For every GT box
        for ii, (label_gt, box_gt) in list(enumerate(zip(labels_gt, boxes_gt))):

          # Consider only boxes with correct label
          if label_gt != label:
            continue

          subject_gt, object_gt = fix_box(box_gt[0]), fix_box(box_gt[1])

          subject_pred, object_pred = get_pred_boxes(box)
          [subject_pred, object_pred] = [fix_box(b)
                                         for b in get_pred_boxes(box)]

          iou1 = IoU(subject_gt, subject_pred)
          iou2 = IoU(object_gt, object_pred)

          if iou1 >= 0.5 and iou2 >= 0.5:
            if correct_detections[ii] < (iou1 * iou2):
              correct_detections[ii] = iou1 * iou2

              ax = f2.add_subplot(1, 2, 1)
              ax.imshow((im[0] * 127.5 + 127.5).astype('uint8'))
              ax.grid('off')
              ax.axis('off')

              ax.add_patch(patches.Rectangle(object_gt[0:2][::-1] * 224,
                                             (object_gt[3] - object_gt[1]) * 224,
                                             (object_gt[2] - object_gt[0]) * 224,
                                             fill=False,
                                             linewidth=7, color='red'))

              ax.add_patch(patches.Rectangle(subject_gt[0:2][::-1] * 224,
                                             (subject_gt[3] - subject_gt[1]) * 224,
                                             (subject_gt[2] - subject_gt[0]) * 224,
                                             fill=False,
                                             linewidth=4, color='blue'))

              ax.set_title("Ground truth for '" + utils.LABEL_MAP[label] + "'")

              ax = f2.add_subplot(1, 2, 2)
              ax.imshow((im[0] * 127.5 + 127.5).astype('uint8'))

              ax.grid('off')
              ax.axis('off')

              ax.add_patch(patches.Rectangle(object_pred[0:2][::-1] * 224,
                                             (object_pred[3] - object_pred[1]) * 224,
                                             (object_pred[2] - object_pred[0]) * 224,
                                             fill=False,
                                             linewidth=7, color='red'))

              ax.add_patch(patches.Rectangle(subject_pred[0:2][::-1] * 224,
                                             (subject_pred[3] - subject_pred[1]) * 224,
                                             (subject_pred[2] - subject_pred[0]) * 224,
                                             fill=False,
                                             linewidth=4, color='blue'))


              ax.set_title("Prediction for '" + utils.LABEL_MAP[label] + "'")

              try:
                os.mkdir('/tmp/results/correct/%s' % utils.LABEL_MAP[label])
              except:
                pass

              f2.subplots_adjust(wspace=0.01, hspace=0.01,
                                left=0.05, right=0.95,
                                bottom=0.05, top=0.95)

              f2.savefig('/tmp/results/correct/%s/%s-%i.jpg' % (utils.LABEL_MAP[label],
                                                               filename[:-4], ii))
              f2.clf()


    box_count += len(labels_gt)
    pos += np.sum(correct_detections > 0)
    print(guesses / float(ex_count + 1))

print(pos / float(box_count))
