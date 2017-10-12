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
import matplotlib.pyplot as plt
import sys
import os
import random
import time
import json
import cPickle
import math

from scipy import io

import matplotlib

from matplotlib import pylab
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from collections import Counter, defaultdict

import tensorflow as tf

from slim.data import dataset_data_provider
from slim.data import dataset
from slim.data import tfexample_decoder
from slim.datasets import datasets
from slim import queues


with open('relationships.json') as f:
    data_rel = json.load(f)


white_list = ['hold', 'rid', 'carry', 'eat', 'watch',
             'look', 'fly', 'swing', 'pull', 'hit', 'touch', 'throw', 'cast',
             'cut', 'read', 'catch', 'talk', 'drink', 'look',
             'swim', 'push', 'feed', 'graze', 'reflect', 'kick',
             'float', 'perch', 'brush', 'reach', 'pet', 'talk', 'serve',
             'sew', 'sniff', 'chase', 'lick', 'swing', 'hug',
             'lift', 'splash', 'spray']

def preprocess_relation(x):

  x = [WordNetLemmatizer().lemmatize(token ,'v')
       for token in x.split(' ')]

  for xx in x:
    if xx in white_list:
      return xx

  return ''

nltk.data.path = ['/home/akolesnikov/nltk_data']

result = {}
for entry in data_rel:
  cur = entry['image_id'], []
  for r in entry['relationships']:
    predicate = preprocess_relation(r['predicate'].strip().lower())
    if predicate:
      try:
        cur[1].append({'predicate': predicate,
                       'object_bbox': (r['object']['y'], r['object']['x'],
                                       r['object']['h'], r['object']['w']),
                       'subject_bbox': (r['subject']['y'], r['subject']['x'],
                                        r['subject']['h'], r['subject']['w']),
                       'object_name': str(r['object']['name']),
                       'subject_name': str(r['subject']['name'])})
      except:
        continue
  if cur[1]:
    result[cur[0]] = cur[1]


predicate_list = [e['predicate'] for d in result.values() for e in d]
[x for x in sorted(Counter(predicate_list).items(), key=lambda x: -x[1])]


class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def read_image_and_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image, image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def int64_feature(values):
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def float_feature(values):
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))


chars = range(ord("a"), ord("z") + 1) + [ord(" "), ord("X")]
ord_map = defaultdict(int)
ord_map.update(dict(zip(chars, range(1, len(chars) + 1))))

def ord_caption(cap):
  if len(cap) < 24:
    cap += "X" * (24 - len(cap))
  cap = cap[:24]
  return str(cap), np.array([ord_map[ord(x)] for x in cap]).astype("int64")


def get_dataset_filename(dataset_dir, split_name, shard_id, num_shards):
  output_filename = 'VG-%s_%05d-of-%05d.tfrecord' % (split_name, shard_id,
                                                     num_shards)
  return os.path.join(dataset_dir, output_filename)

def image_to_tfexample(image_data, shape, num_relations,
                       relations_raw, relations, relations_label, bboxes,
                       objects, subjects):
  return tf.train.Example(features=tf.train.Features(feature={
      'image/encoded': bytes_feature(image_data),
      'image/shape': int64_feature(shape),
      'image/relations/num': int64_feature(num_relations),
      'image/relations/predicates_raw': bytes_feature(relations_raw),
      'image/relations/predicates': int64_feature(relations),
      'image/relations/predicates_label': int64_feature(relations_label),
      'image/relations/bboxes': int64_feature(bboxes),
      'image/relations/objects': int64_feature(objects),
      'image/relations/subjects': int64_feature(subjects),
  }))

d = '/home/akolesnikov/VG/'

def get_image_list(split):
  train_list = set([int(l.strip()[:-4]) for l in
                   open(os.path.join(d, 'image_lists',
                                  'image_%s_list' % split)).readlines()])
  train_files = [os.path.join(d, 'images', str(i) + '.jpg') for i in result.keys()
                 if i in train_list]
  return train_files

train_files = get_image_list('train')
val_files = get_image_list('val')
test_files = get_image_list('test')


len(train_files), len(val_files), len(test_files)


dataset_dir = '/home/akolesnikov/VG/binary/'

for split, image_filenames, num_shards in [('test', test_files, 10),
                                           ('val', val_files, 10),
                                           ('train', train_files, 100)]:

  dataset_pickle = []

  num_images = len(image_filenames)
  num_per_shard = int(math.ceil(num_images / float(num_shards)))

  with tf.Graph().as_default():
    image_reader = ImageReader()

    for shard_id in xrange(num_shards):
      with tf.Session() as sess:
        output_filename = get_dataset_filename(dataset_dir, split, shard_id,
                                               num_shards)
        print("Processing %s" % output_filename)
        print(output_filename)

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
          start_ndx = shard_id * num_per_shard
          end_ndx = min((shard_id+1) * num_per_shard, num_images)
          for i in xrange(start_ndx, end_ndx):
            filename = image_filenames[i]

            # load the image
            image_data = tf.gfile.FastGFile(filename, 'r').read()
            image, height, width = image_reader.read_image_and_dims(sess,
                                                                    image_data)

            image_id = int(os.path.basename(filename)[:-4])
            bboxes = [val
                      for rel in result[image_id]
                      for val in rel['object_bbox'] + rel['subject_bbox']]
            caps_raw = [ord_caption(x['predicate'])[0]
                        for x in result[image_id]]
            caps = [ord_caption(x['predicate'])[1]
                    for x in result[image_id]]
            caps_index = [white_list.index(x['predicate'])
                          for x in result[image_id]]

            objects = [ord_caption(x['object_name'])[1]
                       for x in result[image_id]]
            subjects = [ord_caption(x['subject_name'])[1]
                       for x in result[image_id]]


            num_relations = len(result[image_id])
            caps_raw = str(''.join(caps_raw))
            caps = list(np.hstack(caps))
            objects = list(np.hstack(objects))
            subjects = list(np.hstack(subjects))


            example = image_to_tfexample(image_data, [height, width],
                                         num_relations, caps_raw, caps, caps_index,
                                         bboxes,
                                         objects, subjects)

            dataset_pickle.append([os.path.basename(filename),
                                   caps_index,
                                   bboxes])

            # write to stream
            tfrecord_writer.write(example.SerializeToString())
  cPickle.dump(dataset_pickle, open('/home/akolesnikov/'
                                    'VG/pickle/%s.pickle' % split, 'w'),
               protocol=2)
