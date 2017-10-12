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

import tensorflow as tf

import numpy as np


# Misc utils #########################################


def concat_elu(x):
  return tf.nn.elu(tf.concat([x, -x], axis=3))

# Basic layers #########################################


# High level ops #########################################


def gated_resnet(x, name, nonlinearity=concat_elu, conv=conv2d,
                 a_aux=None, h_aux=None, dilation=None):

  with tf.variable_scope(name):
    num_filters = int(x.get_shape()[-1])

    c1 = conv(nonlinearity(x), "conv1", (3, 3), num_filters, dilation=dilation)

    if a_aux is not None:
      assert(len(a_aux.get_shape().as_list()) == 4)
      a_aux = conv2d(nonlinearity(a_aux), "conv_aux", (1, 1), num_filters)
      a_aux = tf.image.resize_bilinear(a_aux, c1.get_shape()[1:3])
      c1 += a_aux

    c1 = nonlinearity(c1)

    if tf.GLOBAL["dropout"] > 0:
      c1 = tf.nn.dropout(c1, keep_prob=1.0 - tf.GLOBAL["dropout"])

    c2 = conv(c1, "conv2", (3, 3), num_filters * 2, init_scale=0.1)

    if h_aux is not None:
      assert(len(h_aux.get_shape().as_list()) == 2)
      h_aux = conv2d(h_aux[:, None, None, :], "conv_cond", (1, 1),
                     num_filters, init_scale=0.1)
      c2 += nonlinearity(h_aux)

    a, b = tf.split(c2, 2, 3)
    return a * tf.nn.sigmoid(b) + x


def parseImage(x,
               h_aux, a_aux,
               channels_mult, det_channels_mult,
               num_classes_list,
               num_boxes):

  channels = int(64 * channels_mult)
  i_res_block = 0

  with tf.variable_scope("image_parser"):
    # 224 x 224 x 3
    layer = conv2d(x, "conv1", filter_size=(3, 3),
                   stride=2, out_channels=channels)

    # 112 x 112 x (64 * m)
    for _ in range(2):
      i_res_block += 1
      layer = gated_resnet(layer, "res_block_%d" % i_res_block,
                           h_aux=h_aux,
                           conv=conv2d)
    channels *= 2
    layer = conv2d(layer, "downscale_layer_1", filter_size=(3, 3),
                   stride=2, out_channels=channels)

    # 56 x 56 x (128 * m)
    for _ in range(2):
      i_res_block += 1
      layer = gated_resnet(layer, "res_block_%d" % i_res_block,
                           h_aux=h_aux,
                           conv=conv2d)

    layer_detection = layer
    channels *= 2
    layer = conv2d(layer, "downscale_layer_2", filter_size=(3, 3),
                   stride=2, out_channels=channels)

    # 28 x 28 x (256 * m)
    for _ in range(2):
      i_res_block += 1
      layer = gated_resnet(layer, "res_block_%d" % i_res_block,
                           h_aux=h_aux,
                           conv=conv2d)

    channels *= 2
    layer = conv2d(layer, "downscale_layer_3", filter_size=(3, 3),
                   stride=2, out_channels=channels)

    # 14 x 14 x (512 * m)
    for _ in range(2):
      i_res_block += 1
      layer = gated_resnet(layer, "res_block_%d" % i_res_block,
                           h_aux=h_aux,
                           conv=conv2d)

    channels *= 1
    layer = conv2d(layer, "downscale_layer_4", filter_size=(3, 3),
                   stride=2, out_channels=channels)

    # 7 x 7 x (512 * m)
    layer = tf.stop_gradient(layer)
    for _ in range(2):
      i_res_block += 1
      layer = gated_resnet(layer, "res_block_%d" % i_res_block,
                           h_aux=h_aux,
                           conv=conv2d)

    # Prediction layer
    layer = tf.reduce_mean(layer, axis=[1, 2], keep_dims=True)

    logits = []
    for num in num_classes_list:
      logit = conv2d(layer, "predictions%i" % num, filter_size=(1, 1),
                     stride=1, out_channels=num,
                     init_scale=0.0001)[:, 0, 0, :]
      logits.append(logit)

  # detection path

  with tf.variable_scope("detector"):
    channels = int(160 * det_channels_mult)

    layer = tf.concat([layer_detection, a_aux], axis=3)
    layer = conv2d(layer, "adapt", filter_size=(1, 1),
                   stride=1, out_channels=channels)

    i_res_block = 0
    for dilation in [None, 2, 4, 8]:
      for _ in range(2):
        i_res_block += 1
        layer = gated_resnet(layer, "res_block_detection_%d" % i_res_block,
                             a_aux=a_aux,
                             h_aux=h_aux, dilation=dilation)

    point_ps = []
    for i in range(num_boxes):
      point_p = conv2d(layer, "detection%i" % i, filter_size=(1, 1),
                       stride=1, out_channels=1, init_scale=0.0001)
      point_ps.append(point_p)

  return logits, point_ps


def main(argv=()):
  del argv  # Unused.


if __name__ == "__main__":
  pass
