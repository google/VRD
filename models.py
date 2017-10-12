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

import nn


def model_detection(images,
                    labels,
                    boxes,
                    stage,
                    num_classes=41):

  input_mask = tf.to_float(1 - tf.cumsum(tf.one_hot(stage, 5)))
  loss_mask = tf.one_hot(stage, 5)

  labels_aux = tf.concat([label * input_mask[i]
                          for i, label in enumerate(labels)], axis=0)
  boxes_aux = boxes * input_mask[1:][None, None, None, :]

  labels_p, points_p = nn.parseImage(images,
                                     h_aux=labels_aux, a_aux=boxes_aux,
                                     channels_mult=1,
                                     det_channels_mult=1,
                                     num_classes_list=[num_classes],
                                     num_boxes=4)

  # Get label losses
  losses_label = [tf.reduce_sum(
      tf.nn.softmax_cross_entropy_with_logits(logits=l_p, labels=l_gt))
                  for l_p, l_gt in zip(labels_p, labels)]

  losses_label = tf.stack(losses_label)

  # Get box losses
  points_p_norm = tf.concat(points_p, axis=3)
  points_p_norm -= tf.reduce_logsumexp(points_p_norm,
                                       axis=[1, 2],
                                       keep_dims=True)
  losses_box = -tf.reduce_sum(points_p_norm * boxes, axis=[0, 1, 2])

  # Merge losses and select loss from a proper stage
  losses = tf.concat([losses_label, losses_box], axis=0)
  loss = tf.reduce_sum(losses * loss_mask)
  loss /= (tf.to_float(tf.shape(images)[0]) * tf.log(2.0))

  return [labels_p, points_p], loss


def main(argv=()):
  del argv  # Unused.


if __name__ == "__main__":
  pass
