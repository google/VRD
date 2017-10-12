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

from slim import queues

import time

import nn
import models
import utils


# Global params ############################

flags.DEFINE_string("tb_log_dir",
                    "./log",
                    "Tensorboard log dir")

flags.DEFINE_enum("mode",
                  "gpu",
                  ["cpu", "gpu"],
                  "Specify computation mode: CPU or GPU")

flags.DEFINE_integer("num_gpus",
                     1,
                     "Number of GPUs")

flags.DEFINE_integer("run_test",
                     0,
                     "Only run model evaluation")

# Optimization params ############################
flags.DEFINE_integer("batch_size_per_gpu",
                     8,
                     "Batch size per GPU")

flags.DEFINE_integer("num_epochs",
                     1000,
                     "Number of epochs")

flags.DEFINE_integer("init_batch_size",
                     64,
                     "Batch size for data-dependent initialiaztion")

flags.DEFINE_integer("iter_cap",
                     20,
                     "Cap on the number of training steps per epoch")

flags.DEFINE_float("decay",
                   0.99997,
                   "Learning rate decay rate")

flags.DEFINE_integer("use_pretrained",
                     1,
                     "Use pretrained")

# Modeling parameters ############################

flags.DEFINE_integer("image_res",
                     224,
                     "Number of channels")

# Log parameters ############################

flags.DEFINE_integer("log_training_loss",
                     20,
                     "How often to log training loss")

flags.DEFINE_integer("log_val_loss",
                     100,
                     "How often to log val loss")

FLAGS = flags.FLAGS

# External data ##################################


def main(argv=()):
  del argv

  batch_size = FLAGS.batch_size_per_gpu * FLAGS.num_gpus

  data_stream_init = utils.setup_data_stream_genome("train",
                                                    batch_size=FLAGS.init_batch_size,
                                                    image_res=FLAGS.image_res,
                                                    )
  (image_init_batch, class_init_batch,
   box_init_batch) = data_stream_init

  data_stream_train = utils.setup_data_stream_genome("train",
                                                     batch_size=batch_size,
                                                     image_res=FLAGS.image_res)
  (image_train_batch, class_train_batch,
   box_train_batch) = data_stream_train

  data_stream_val = utils.setup_data_stream_genome("val",
                                                   batch_size=batch_size,
                                                   image_res=FLAGS.image_res)
  (image_val_batch, class_val_batch,
   box_val_batch) = data_stream_val

  def model_template(images, labels,
                     boxes,
                     stage):
    return models.model_detection(images, labels,
                                  boxes,
                                  stage)

  model_factory = tf.make_template("detection", model_template)

  tf.GLOBAL = {}

  # Init
  tf.GLOBAL["init"] = True
  tf.GLOBAL["dropout"] = 0.0

  with tf.device("/cpu:0"):
    _ = model_factory(image_init_batch, [class_init_batch],
                      box_init_batch,
                      0)
  ## Train
  tf.GLOBAL["init"] = False
  tf.GLOBAL["dropout"] = 0.5

  imgs_train = tf.split(image_train_batch, FLAGS.num_gpus, 0)
  class_train = tf.split(class_train_batch, FLAGS.num_gpus, 0)
  boxes_train = tf.split(box_train_batch, FLAGS.num_gpus, 0)

  min_stage = tf.placeholder(shape=[], dtype=tf.int32)
  stage_train = tf.random_uniform([], min_stage, 5, dtype=tf.int32)

  loss_train = 0.0
  for i in range(FLAGS.num_gpus):
    with tf.device("gpu:%i" % i if FLAGS.mode == "gpu" else "/cpu:0"):
      _, loss = model_factory(imgs_train[i],
                              [class_train[i]],
                              boxes_train[i],
                              stage_train)
      loss_train = loss_train + loss

  loss_train /= FLAGS.num_gpus

  # Optimization
  learning_rate = tf.Variable(0.0001)
  update_lr = learning_rate.assign(FLAGS.decay * learning_rate)
  optimizer = tf.train.AdamOptimizer(learning_rate, 0.95, 0.9995)
  train_step = optimizer.minimize(loss_train,
                                  colocate_gradients_with_ops=True)

  train_bpd_ph = tf.placeholder(shape=[], dtype=tf.float32)
  summary_train = {i: tf.summary.scalar("train_bpd_stage%i" % i, train_bpd_ph)
                   for i in range(5)}

  ## Val
  tf.GLOBAL["init"] = False
  tf.GLOBAL["dropout"] = 0.0

  imgs_val = tf.split(image_val_batch, FLAGS.num_gpus, 0)
  class_val = tf.split(class_val_batch, FLAGS.num_gpus, 0)
  boxes_val = tf.split(box_val_batch, FLAGS.num_gpus, 0)
  stage_val = tf.random_uniform([], 0, 5, dtype=tf.int32)

  loss_val = 0.0
  label_p_val, point_p_val = [], []
  for i in range(FLAGS.num_gpus):
    with tf.device("gpu:%i" % i if FLAGS.mode == "gpu" else "/cpu:0"):
      [label_p_v, point_p_v], loss = model_factory(imgs_val[i],
                                                   [class_val[i]],
                                                   boxes_val[i],
                                                   stage_val)
      loss_val = loss_val + loss
      label_p_val.append(label_p_v)
      point_p_val.append(point_p_v)

  loss_val /= FLAGS.num_gpus
  label_p_val = [tf.concat(l, axis=0) for l in zip(*label_p_val)]
  point_p_val = [tf.concat(l, axis=0) for l in zip(*point_p_val)]

  val_bpd_ph = tf.placeholder(shape=[], dtype=tf.float32)
  summary_val = {i: tf.summary.scalar("val_bpd_stage%i" % i, val_bpd_ph)
                 for i in range(5)}

  # Counters
  global_step, val_step = tf.Variable(1), tf.Variable(1)
  update_global_step = global_step.assign_add(1)
  update_val_step = val_step.assign_add(1)

  ## Inits
  var_init_1 = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                if v.name.find("image_parser") >= 0]
  var_init_2 = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                if v.name.find("detector") >= 0]
  var_rest = list(set(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)) -
                  set(var_init_1 + var_init_2))

  init_ops = [tf.initialize_variables(v_l) for v_l in
              [var_init_1, var_init_2, var_rest]]

  ####
  image_summary_placeholder = tf.placeholder(dtype=tf.float32)
  image_summary_sample_val = tf.summary.image("validation_samples",
                                              image_summary_placeholder,
                                              max_outputs=256)
  saver = tf.train.Saver()

  # tf.get_default_graph().finalize()
  with tf.Session() as sess:
    with queues.QueueRunners(sess):

      default_model_meta = os.path.join(FLAGS.tb_log_dir,
                                        "main", "model.ckpt.meta")
      default_model_file = os.path.join(FLAGS.tb_log_dir,
                                        "main", "model.ckpt")
      rerun = False
      if tf.gfile.Exists(default_model_meta):
        print("Model is loading...")
        saver.restore(sess, default_model_file)
        rerun = True
      else:
        # Initialization (Due to the bug in tensorflow it is split
        #                 into multiple steps)
        _ = [sess.run(init_op) for init_op in init_ops]

        if FLAGS.use_pretrained:
          utils.optimistic_restore(sess, "")
          sess.run(global_step.assign(1))
          sess.run(val_step.assign(1))
          sess.run(learning_rate.assign(0.0001))

      # Summary writers
      summary_writer_main = tf.summary.FileWriter("%s/%s" % (FLAGS.tb_log_dir,
                                                             "main"),
                                                  sess.graph)
       # Visalize validation GT

      if not rerun:
        (imgs_sample, box_cls_sample,
         boxes_sample) = sess.run([image_val_batch, class_val_batch,
                                   box_val_batch])

        boxes_sample = np.concatenate([boxes_sample[..., :2][:, None],
                                       boxes_sample[..., 2:][:, None]], 1)
        imgs_with_box = utils.visualize(imgs_sample, box_cls_sample,
                                        boxes_sample, utils.LABEL_MAP)

        s = sess.run(image_summary_sample_val,
                     {image_summary_placeholder: np.array(imgs_with_box)})
        summary_writer_main.add_summary(s, 0)

     # Run training

      n_iter_train = (utils.SST_COUNTS["train"] // batch_size
                      if FLAGS.iter_cap <= 0
                      else FLAGS.iter_cap)
      n_iter_val = (utils.SST_COUNTS["val"] // batch_size
                    if FLAGS.iter_cap <= 0
                    else FLAGS.iter_cap)

      max_iter = FLAGS.num_epochs * n_iter_train

      buf_loss = defaultdict(list)
      val_i = 0
      while True and (not FLAGS.run_test):

       # Training step
        (_, loss_v, stage_v,
         train_i, val_i) = sess.run([train_step,
                                     loss_train, stage_train,
                                     global_step, val_step],
                                     {min_stage: 0})

        buf_loss[stage_v].append(loss_v)

        # Update global counter and learning rate
        sess.run([update_global_step, update_lr])

        # Log training error
        if train_i % FLAGS.log_training_loss == 0:

          for i in range(5):
            s = sess.run(summary_train[i], {train_bpd_ph: np.mean(buf_loss[i])})
            summary_writer_main.add_summary(s, train_i)
          buf_loss = defaultdict(list)

        # Log val error and visualize samples
        if train_i % FLAGS.log_val_loss == 0:

          buf_loss = defaultdict(list)
          for i in range(n_iter_val):
            loss_v, stage_v = sess.run([loss_val, stage_val])
            buf_loss[stage_v].append(loss_v)

          for i in range(5):
            s = sess.run(summary_val[i], {val_bpd_ph: np.mean(buf_loss[i])})
            summary_writer_main.add_summary(s, val_i)
          buf_loss = defaultdict(list)

          # Sample detections

          label_np = np.zeros((batch_size, 41))
          boxes_np = np.zeros((batch_size, 56, 56, 4))

          # stage 0
          l = sess.run(label_p_val, {image_val_batch: imgs_sample,
                                     class_val_batch: label_np,
                                     box_val_batch: boxes_np,
                                     stage_val: 0})[0]
          l = np.argmax(l, axis=1)
          label_np[range(batch_size), l] = 1

          # stage 1
          for ii in range(4):
            l = sess.run(point_p_val, {image_val_batch: imgs_sample,
                                       class_val_batch: label_np,
                                       box_val_batch: boxes_np,
                                       stage_val: ii + 1})[ii]
            l = (l == np.amax(l, axis=(1, 2), keepdims=True)).astype("int32")
            boxes_np[:, :, :, ii:ii + 1] = l

          # vis
          boxes_np = np.concatenate([boxes_np[..., :2][:, None],
                                     boxes_np[..., 2:][:, None]], 1)
          imgs_with_box = utils.visualize(imgs_sample, label_np,
                                          boxes_np, utils.LABEL_MAP)

          image_summary_det = tf.summary.image("detection_samples%i" % val_i,
                                               image_summary_placeholder,
                                               max_outputs=256)

          s = sess.run(image_summary_det,
                       {image_summary_placeholder: np.array(imgs_with_box)})
          summary_writer_main.add_summary(s, 0)

          # Save model
          saver.save(sess, os.path.join(FLAGS.tb_log_dir,
                                        "main",
                                        "model.ckpt"))
          saver.save(sess, os.path.join(FLAGS.tb_log_dir,
                                        "main",
                                        "model%i.ckpt" % val_i))

          sess.run([update_val_step])

        # Terminate
        if train_i > max_iter:
          break

      if FLAGS.run_test:
        pass


if __name__ == "__main__":
  pass
