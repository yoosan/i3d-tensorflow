from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools

import tensorflow as tf
import i3d, i3d_v2, r3d

FLAGS = tf.flags.FLAGS

networks_map = {'i3d_v1': i3d.I3D,
                'i3d_v2': i3d_v2.I3D_V2,
                'r3d_50': r3d.resnet_v1_50,
                'r3d_101': r3d.resnet_v1_101,
                'r3d_152': r3d.resnet_v1_152
               }

def get_network_fn(name, num_classes, weight_decay=0.0, is_training=False, data_format='NHWC'):
  """Returns a network_fn such as `logits, end_points = network_fn(images)`.

  Args:
    name: The name of the network.
    num_classes: The number of classes to use for classification.
    weight_decay: The l2 coefficient for the model weights.
    is_training: `True` if the model is being used for training and `False`
      otherwise.

  Returns:
    network_fn: A function that applies the model to a batch of images. It has
      the following signature:
        logits, end_points = network_fn(images)
  Raises:
    ValueError: If network `name` is not recognized.
  """
  if name not in networks_map:
    raise ValueError('Name of network unknown %s' % name)
  func = networks_map[name]
  trainBN = (not FLAGS.freezeBN) and is_training
  @functools.wraps(func)
  def network_fn(images):
    return func(images, num_classes=num_classes, is_training=is_training,
                final_endpoint='Predictions', data_format=data_format, dropout_keep_prob=FLAGS.dropout_keep)
  if hasattr(func, 'default_image_size'):
    network_fn.default_image_size = func.default_image_size

  return network_fn