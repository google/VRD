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


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np

import tensorflow as tf

from slim.data import dataset
from slim.data import dataset_data_provider
from slim.data import tfexample_decoder
from slim.datasets import datasets

OPENIMAGES_COUNTS = {'train': 1247361,
                     'val': 95181}

SST_COUNTS = {'train': 32649,
              'val': 3737}

LABEL_MAP = {i: s for i, s in enumerate(
    ['hold', 'rid', 'carry', 'eat', 'watch',
     'look', 'fly', 'swing', 'pull', 'hit', 'touch', 'throw', 'cast',
     'cut', 'read', 'catch', 'talk', 'drink', 'look',
     'swim', 'push', 'feed', 'graze', 'reflect', 'kick',
     'float', 'perch', 'brush', 'reach', 'pet', 'talk', 'serve',
     'sew', 'sniff', 'chase', 'lick', 'swing', 'hug',
     'lift', 'splash', 'spray']
    )}


def get_corners(rbox):
  ul_point = (tf.one_hot(tf.to_int32(rbox[0] * 55), 56)[:, None] *
              tf.one_hot(tf.to_int32(rbox[1] * 55), 56)[None, :])
  lr_point = (tf.one_hot(tf.to_int32(rbox[2] * 55), 56)[:, None] *
              tf.one_hot(tf.to_int32(rbox[3] * 55), 56)[None, :])
  box = tf.to_float(tf.concat([ul_point[:, :, None], lr_point[:, :, None]], 2))

  return box


def get_split_genome(name, root_dir, reader_class=tf.TFRecordReader):
  """Produces data split.

  Args:
    name: split name
    root_dir: dataset directory
    reader_class: reader class

  Returns:
    Dataset
  """

  file_pattern = os.path.join(root_dir, 'VG-%s_*' % name)

  num_samples = SST_COUNTS[name]

  keys_to_features = {
      'image/encoded': tf.FixedLenFeature(
          (), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature(
          (), tf.string, default_value='jpeg'),
      'image/shape': tf.VarLenFeature(dtype=tf.int64),
      'image/relations/num': tf.VarLenFeature(dtype=tf.int64),
      'image/relations/predicates_raw': tf.FixedLenFeature((), tf.string),
      'image/relations/predicates': tf.VarLenFeature(dtype=tf.int64),
      'image/relations/predicates_label': tf.VarLenFeature(dtype=tf.int64),
      'image/relations/bboxes': tf.VarLenFeature(dtype=tf.int64),
      'image/relations/objects': tf.VarLenFeature(dtype=tf.int64),
      'image/relations/subjects': tf.VarLenFeature(dtype=tf.int64),
  }

  items_to_handlers = {
      'image': tfexample_decoder.Image('image/encoded'),
      'shape': tfexample_decoder.Tensor('image/shape'),
      'num_relations': tfexample_decoder.Tensor('image/relations/num'),
      'predicates_raw': tfexample_decoder.Tensor('image/relations/'
                                                 'predicates_raw'),
      'predicates': tfexample_decoder.Tensor('image/relations/predicates'),
      'predicates_label': tfexample_decoder.Tensor('image/relations/predicates_label'),
      'bboxes': tfexample_decoder.Tensor('image/relations/bboxes'),
      'objects': tfexample_decoder.Tensor('image/relations/objects'),
      'subjects': tfexample_decoder.Tensor('image/relations/subjects')
  }

  decoder = tfexample_decoder.TFExampleDecoder(keys_to_features,
                                               items_to_handlers)

  return dataset.Dataset(
      data_sources=file_pattern,
      reader=reader_class,
      decoder=decoder,
      num_samples=num_samples,
      items_to_descriptions={})


def setup_data_stream_genome(name,
                             batch_size,
                             image_res,
                             shuffle=True,
                             root_dir=):
  """Setup data stream for visual genome.

  Args:
    name: split name
    root_dir: dataset home
    batch_size: batch size
    num_classes: number of classes
    shuffle: shuffle data

  Returns:
    image, raw segmetation, downsampled segmentation,
    raw caption and caption tensroflow streams
  """

  data = get_split_genome(name, root_dir)
  data_provider = dataset_data_provider.DatasetDataProvider(data,
                                                            num_readers=4,
                                                            shuffle=shuffle)
  (image, shape, num_relations,
   predicates, bboxes, obj, subj) = data_provider.get(['image',
                                                       'shape',
                                                       'num_relations',
                                                       'predicates_label',
                                                       'bboxes',
                                                       'objects',
                                                       'subjects'])
  # Fix random relation
  num_relations = num_relations[0]
  i_rel = tf.random_uniform([], 0, num_relations, dtype=tf.int64)

  # Resize image
  image = tf.image.resize_bilinear(image[None], [image_res, image_res])[0]
  image = (image - 127.5) / 127.5

  predicates = tf.one_hot(predicates, 41)

  # Produce target segmentation
  shape = tf.to_float(shape)
  b = tf.to_float(bboxes)
  b = tf.reshape(tf.reshape(b, [-1, 8])[i_rel], [2, 4])
  b = tf.concat([b[:, 0:1] / shape[0], b[:, 1:2] / shape[1],
                 (b[:, 0:1] + b[:, 2:3]) / shape[0],
                 (b[:, 1:2] + b[:, 3:4]) / shape[1]], axis=1)

  b_rev = tf.concat([b[:, 0:1], 1 - b[:, 3:4],
                     b[:, 2:3], 1 - b[:, 1:2]], axis=1)

  boxes = tf.concat([get_corners(b[1]), get_corners(b[0])], 2)
  boxes_rev = tf.concat([get_corners(b_rev[1]), get_corners(b_rev[0])], 2)

  is_flip = tf.equal(tf.random_uniform([], 0, 2, dtype=tf.int32), 1)
  image = tf.cond(is_flip, lambda: tf.reverse(image, [1]), lambda: image)
  boxes = tf.cond(is_flip, lambda: boxes_rev, lambda: boxes)

  # Batch data
  (image, preds, boxes) = tf.train.batch([image, predicates[i_rel], boxes],
                                         batch_size, num_threads=4)

  return image, preds, boxes


### MISC #####################################


def visualize(imgs, box_cls, boxes, lmap):

  s = float(boxes.shape[2])

  imgs_with_box = []
  for img, c, box in zip(imgs, box_cls, boxes):

    box = [np.array([np.where(b[..., 0])[0][0] / s,
                     np.where(b[..., 0])[1][0] / s,
                     np.where(b[..., 1])[0][0] / s,
                     np.where(b[..., 1])[1][0] / s])
           for b in box]
    box = np.array(box)

    img = (img * 127.5 + 127.5).astype('uint8')
    vis_util.visualize_boxes_and_labels_on_image_array(
        img,
        box,
        np.array([np.where(c)[0][0]] * box.shape[0]),
        np.array([1] * box.shape[0]),
        {k: {'name': v} for k, v in lmap.items()},
        use_normalized_coordinates=True)
    imgs_with_box.append(img)

  return np.array(imgs_with_box)


def tile_image(x_gen, tiles=()):
  """Tiled image representations.

  Args:
    x_gen: 4D array of images (n x w x h x 3)
    tiles (int pair, optional): number of rows and columns

  Returns:
    Array of tiled images (1 x W x H x 3)
  """
  n_images = x_gen.shape[0]
  if not tiles:
    for i in range(int(np.sqrt(n_images)), 0, -1):
      if n_images % i == 0:
        break
    n_rows = i
    n_cols = n_images // i
  else:
    n_rows, n_cols = tiles
  full = [np.hstack(x_gen[c * n_rows:(c + 1) * n_rows]) for c in range(n_cols)]
  return np.expand_dims(np.vstack(full), 0)


def main(argv=()):
  del argv  # Unused.


if __name__ == '__main__':
  pass
