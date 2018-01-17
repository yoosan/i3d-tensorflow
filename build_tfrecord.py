"""Converts video data to TFRecords file format with Example protos.

The video data set is expected to reside in avi/mp4 files located in the
following directory structure.

  data_dir/label_0/video0.avi
  data_dir/label_0/video1.avi
  ...UCF101 example
  ucf101_dir/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi
  ...

where the sub-directory is the unique label associated with these videos.

This TensorFlow script converts the training and evaluation data into
a sharded data set consisting of TFRecord files

  train_directory/train-00000-of-01024
  train_directory/train-00001-of-01024
  ...
  train_directory/train-01023-of-01024

and

  validation_directory/validation-00000-of-00128
  validation_directory/validation-00001-of-00128
  ...
  validation_directory/validation-00127-of-00128

Reading video using skivideo package
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import random
import math
import threading
import skvideo.io
from datetime import datetime
import numpy as np
import tensorflow as tf

tf.app.flags.DEFINE_string('train_directory', '/tmp/',
                           'Training data directory')
tf.app.flags.DEFINE_string('validation_directory', '/tmp/',
                           'Validation data directory')
tf.app.flags.DEFINE_string('output_directory', '/tmp/',
                           'Output data directory')

tf.app.flags.DEFINE_integer('train_shards', 2,
                            'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_integer('validation_shards', 2,
                            'Number of shards in validation TFRecord files.')

tf.app.flags.DEFINE_integer('num_threads', 2,
                            'Number of threads to preprocess the images.')

# The labels file contains a list of valid labels are held in this file.
# Assumes that the file contains entries as such:
#   dog
#   cat
#   flower
# where each line corresponds to a label. We map each label contained in
# the file to an integer corresponding to the line number starting from 0.
tf.app.flags.DEFINE_string('labels_file', '', 'Labels file')


FLAGS = tf.app.flags.FLAGS


def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(filename, video_buffer, label, height, width):
  """Build an Example proto for an example.

  Args:
    filename: string, path to an image file, e.g., '/path/to/example.JPG'
    image_buffer: string, JPEG encoding of RGB image
    label: integer, identifier for the ground truth for the network
    text: string, unique human-readable, e.g. 'dog'
    height: integer, image height in pixels
    width: integer, image width in pixels
  Returns:
    Example proto
  """

  colorspace = 'RGB'
  channels = 3
  image_format = 'JPEG'

  example = tf.train.SequenceExample(
        context=tf.train.Features(
            feature={
              'height': _int64_feature(height),
              'width': _int64_feature(width),
              'channels': _int64_feature(channels),
              'labels': _int64_feature(label),
              'video_id': _bytes_feature(tf.compat.as_bytes(os.path.basename(filename)))
        }),
        feature_lists=tf.train.FeatureLists(
            feature_list={
            "data":tf.train.FeatureList(feature=[_bytes_feature(tf.compat.as_bytes(video_buffer[i])) for i in range(len(video_buffer))])
            }
        )
    )
  return example


class ImageCoder(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Create a single Session to run all image coding calls.
    self._sess = tf.Session()

    # Initializes function that decodes RGB JPEG data.
    self._encode_jpeg_data = tf.placeholder(dtype=tf.uint8)
    self._encode_jpeg = tf.image.encode_jpeg(self._encode_jpeg_data)

  def encode_jpeg(self, image_data):
    image = self._sess.run(self._encode_jpeg,
                           feed_dict={self._encode_jpeg_data: image_data})
    return image


def _process_video(filename, coder):
  """Process a single video file.

  Args:
    filename: string, path to an image file e.g., '/path/to/example.avi'.
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
  Returns:
    frames: a collection of RGB frames.
    height: integer, image height in pixels.
    width: integer, image width in pixels.
  """
  # Read the image file.
  vid_in = skvideo.io.FFmpegReader(filename)
  data = skvideo.io.ffprobe(filename)['video']
  fps = data['@r_frame_rate'].split('/')
  fps = math.ceil(float(fps[0])/float(fps[1]))
  width = int(data['@width'])
  height = int(data['@height'])
  vid_in._close()

  frames = []
  for idx, frame in enumerate(vid_in.nextFrame()):
    if (idx % (fps)) == 0:
      frame = coder.encode_jpeg(frame)
      frames.append(frame)

  return frames, height, width


def _process_video_files_batch(coder, thread_index, ranges, name, filenames,
                               labels, num_shards):
  """Processes and saves list of videos as TFRecord in 1 thread.

  Args:
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
    thread_index: integer, unique batch to run index is within [0, len(ranges)).
    ranges: list of pairs of integers specifying ranges of each batches to
      analyze in parallel.
    name: string, unique identifier specifying the data set
    filenames: list of strings; each string is a path to an image file
    texts: list of strings; each string is human readable, e.g. 'dog'
    labels: list of integer; each integer identifies the ground truth
    num_shards: integer number of shards for this data set.
  """
  # Each thread produces N shards where N = int(num_shards / num_threads).
  # For instance, if num_shards = 128, and the num_threads = 2, then the first
  # thread would produce shards [0, 64).
  num_threads = len(ranges)
  assert not num_shards % num_threads
  num_shards_per_batch = int(num_shards / num_threads)

  shard_ranges = np.linspace(ranges[thread_index][0],
                             ranges[thread_index][1],
                             num_shards_per_batch + 1).astype(int)
  num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

  counter = 0
  error_cnt = 0
  for s in range(num_shards_per_batch):
    # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
    shard = thread_index * num_shards_per_batch + s
    output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
    output_file = os.path.join(FLAGS.output_directory, output_filename)
    writer = tf.python_io.TFRecordWriter(output_file)

    shard_counter = 0
    files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
    for i in files_in_shard:
      filename = filenames[i]
      label = labels[i]

      try:
        video_buffer, height, width = _process_video(filename, coder)

      except Exception as e:
        print(e)
        print('SKIPPED: Unexpected eror while decoding %s.' % filename)
        continue

      example = _convert_to_example(filename, video_buffer, label,
                                    height, width)
      writer.write(example.SerializeToString())
      shard_counter += 1
      counter += 1

      if not counter % 1000:
        print('%s [thread %d]: Processed %d of %d images in thread batch.' %
              (datetime.now(), thread_index, counter, num_files_in_thread))
        sys.stdout.flush()

    writer.close()
    print('%s [thread %d]: Wrote %d images to %s' %
          (datetime.now(), thread_index, shard_counter, output_file))
    sys.stdout.flush()
    shard_counter = 0
  print('%s [thread %d]: Wrote %d images to %d shards.' %
        (datetime.now(), thread_index, counter, num_files_in_thread))
  print('%s count of error file %d' % (datetime.now(), error_cnt))
  sys.stdout.flush()


def _process_video_files(name, filenames, labels, num_shards):
  """Process and save list of images as TFRecord of Example protos.

  Args:
    name: string, unique identifier specifying the data set
    filenames: list of strings; each string is a path to an image file
    texts: list of strings; each string is human readable, e.g. 'dog'
    labels: list of integer; each integer identifies the ground truth
    num_shards: integer number of shards for this data set.
  """
  assert len(filenames) == len(labels)

  # Break all images into batches with a [ranges[i][0], ranges[i][1]].
  spacing = np.linspace(0, len(filenames), FLAGS.num_threads + 1).astype(np.int)
  ranges = []
  for i in range(len(spacing) - 1):
    ranges.append([spacing[i], spacing[i + 1]])

  # Launch a thread for each batch.
  print('Launching %d threads for spacings: %s' % (FLAGS.num_threads, ranges))
  sys.stdout.flush()

  # Create a mechanism for monitoring when all threads are finished.
  coord = tf.train.Coordinator()

  # Create a generic TensorFlow-based utility for converting all image codings.
  coder = ImageCoder()

  threads = []
  for thread_index in range(len(ranges)):
    args = (coder, thread_index, ranges, name, filenames,
            labels, num_shards)
    t = threading.Thread(target=_process_video_files_batch, args=args)
    threads.append(t)
  
  for t in threads:
    t.setDaemon(True)
    t.start()

  # Wait for all the threads to terminate.
  coord.join(threads)
  print('%s: Finished writing all %d images in data set.' %
        (datetime.now(), len(filenames)))
  sys.stdout.flush()


def _find_video_files(data_dir, labels_file):
  """Build a list of all videos files and labels in the data set.

  Args:
    data_dir: string, path to the root directory of videos.

      Assumes that the image data set resides in video files located in
      the following directory structure.

        ucf101_dir/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi

      where 'ApplyEyeMakeup' is the label associated with these videos.

    labels_file: string, path to the labels file.

  Returns:
    filenames: list of strings; each string is a path to a video file.
    texts: list of strings; each string is the class, e.g. 'ApplyEyeMakeup'
    labels: list of integer; each integer identifies the ground truth.
  """
  print('Determining list of input files and labels from %s.' % data_dir)

  labels = []
  filenames = []

  with open(labels_file, 'r') as f:
    f.readline()
    label2idx = {}
    for line in f:
      line = line.split(',')
      label2idx[line[0]] = line[1]
    f.close()

  list_files = os.listdir(data_dir)
  for label_str in list_files:
    list_videos = os.listdir(data_dir + '/' + label_str)
    for v in list_videos:
      v = str(v)
      if not v.endswith('.mp4'):
        continue
      label = int(label2idx[label_str])
      filenames.append(data_dir + '/' + label_str + '/' + v)
      labels.append(int(label))

  shuffled_index = list(range(len(filenames)))
  random.seed(12345)
  random.shuffle(shuffled_index)

  filenames = [filenames[i] for i in shuffled_index]
  labels = [labels[i] for i in shuffled_index]

  print('Found %d video files inside %s.' %
        (len(filenames), data_dir))
  return filenames, labels


def _process_dataset(name, directory, num_shards, labels_file):
  """Process a complete data set and save it as a TFRecord.

  Args:
    name: string, unique identifier specifying the data set.
    directory: string, root path to the data set.
    num_shards: integer number of shards for this data set.
    labels_file: string, path to the labels file.
  """
  filenames, labels = _find_video_files(directory, labels_file)
  _process_video_files(name, filenames, labels, num_shards)


def main(unused_argv):
  assert not FLAGS.train_shards % FLAGS.num_threads, (
      'Please make the FLAGS.num_threads commensurate with FLAGS.train_shards')
  assert not FLAGS.validation_shards % FLAGS.num_threads, (
      'Please make the FLAGS.num_threads commensurate with '
      'FLAGS.validation_shards')
  if not os.path.exists(FLAGS.output_directory):
    os.makedirs(FLAGS.output_directory)
  print('Saving results to %s' % FLAGS.output_directory)

  # Run it!
  _process_dataset('train', FLAGS.train_directory,
                   FLAGS.train_shards, FLAGS.labels_file)


if __name__ == '__main__':
  tf.app.run()
