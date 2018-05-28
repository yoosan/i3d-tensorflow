"""Utilities for building Inflated 3D ConvNets """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import collections
from snets.scopes import *
slim = tf.contrib.slim

@add_arg_scope
def unit3D(inputs, output_channels,
           kernel_shape=(1, 1, 1),
           strides=(1, 1, 1),
           activation_fn=tf.nn.relu,
           use_batch_norm=True,
           use_bias=False,
           padding='same',
           is_training=True,
           name=None):
  """Basic unit containing Conv3D + BatchNorm + non-linearity."""
  with tf.variable_scope(name, 'unit3D', [inputs]):
    net = tf.layers.conv3d(inputs, filters=output_channels,
                            kernel_size=kernel_shape,
                            strides=strides,
                            padding=padding,
                            use_bias=use_bias)
    if use_batch_norm:
        net = tf.contrib.layers.batch_norm(net, is_training=is_training)
    if activation_fn is not None:
        net = activation_fn(net)
  return net

@add_arg_scope
def sep3D(inputs, output_channels,
          kernel_shape=(1, 1, 1),
          strides=(1, 1, 1),
          activation_fn=tf.nn.relu,
          use_batch_norm=True,
          use_bias=False,
          padding='same',
          is_training=True,
          name=None):
  """Basic Sep-Conv3D layer with BatchNorm + non-linearity.
  A (k_t, k, k) kernel is replaced by a (1, k, k) kernel and a (k_t, 1, 1) kernel
  """
  k_t, k_h, k_w = kernel_shape
  if type(strides) == int:
    s_t, s_h, s_w = strides, strides, strides
  else:
    s_t, s_h, s_w = strides
  spatial_kernel = (1, k_h, k_w)
  spatial_stride = (1, s_h, s_w)
  temporal_kernel = (k_t, 1, 1)
  temporal_stride = (s_t, 1, 1)
  with tf.variable_scope(name, 'sep3D', [inputs]):
    spatial_net = tf.layers.conv3d(inputs, filters=output_channels,
                                  kernel_size=spatial_kernel,
                                  strides=spatial_stride,
                                  padding=padding,
                                  use_bias=use_bias)
    if use_batch_norm:
        spatial_net = tf.contrib.layers.batch_norm(spatial_net, is_training=is_training)
    if activation_fn is not None:
        spatial_net = activation_fn(spatial_net)
    temporal_net = tf.layers.conv3d(spatial_net, filters=output_channels,
                                  kernel_size=temporal_kernel,
                                  strides=temporal_stride,
                                  padding=padding,
                                  use_bias=use_bias)
    if use_batch_norm:
        temporal_net = tf.contrib.layers.batch_norm(temporal_net, is_training=is_training)
    if activation_fn is not None:
        net = activation_fn(temporal_net)
  return net


@add_arg_scope
def unit3D_same(inputs, output_channels,
        kernel_shape=(1, 1, 1),
        strides=(1, 1, 1),
        activation_fn=tf.nn.relu,
        use_batch_norm=True,
        use_bias=False,
        padding='same',
        rate=1,
        is_training=True,
        name=None):
    if (1, 1, 1) == strides:
        return unit3D(inputs, output_channels, kernel_shape,strides,
            padding=padding, name=name)
    else:
        kernel_size_effective = kernel_shape + (kernel_shape - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        padding = [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]]
        inputs = tf.pad(inputs,padding)
        return unit3D(inputs, output_channels, kernel_shape,
            strides, padding='VALID', name=name)


def subsample3D(inputs, factor, name=None):
  """Subsamples the input along the spatial dimensions.
  Args:
    inputs: A `Tensor` of size [batch, height_in, width_in, channels].
    factor: The subsampling factor.
    scope: Optional variable_scope.

  Returns:
    output: A `Tensor` of size [batch, height_out, width_out, channels] with the
    input, either intact (if factor == 1) or subsampled (if factor > 1).
  """
  if factor == 1:
    return inputs
  else:
    return tf.nn.max_pool3d(inputs, [1, 1, 1, 1, 1],
          strides=[1, factor, factor, factor, 1],
          padding='SAME', name=name)


class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
    """A named tuple describing a ResNet block.

    Its parts are:
        scope: The scope of the `Block`.
        unit_fn: The ResNet unit function which takes as input a `Tensor` and
            returns another `Tensor` with the output of the ResNet unit.
        args: A list of length equal to the number of units in the `Block`. The list
            contains one (depth, depth_bottleneck, stride) tuple for each unit in the
            block to serve as argument to unit_fn.
    """

@add_arg_scope
def bottleneck3D(inputs, depth, depth_bottleneck, stride, rate=1,
         outputs_collections=None, scope=None):
  """Bottleneck residual unit variant with BN after convolutions.

  This is the original residual unit proposed in [1]. See Fig. 1(a) of [2] for
  its definition. Note that we use here the bottleneck variant which has an
  extra bottleneck layer.

  When putting together two consecutive ResNet blocks that use this unit, one
  should use stride = 2 in the last unit of the first block.

  Args:
    inputs: A tensor of size [batch, height, width, channels].
    depth: The depth of the ResNet unit output.
    depth_bottleneck: The depth of the bottleneck layers.
    stride: The ResNet unit's stride. Determines the amount of downsampling of
    the units output compared to its input.
    rate: An integer, rate for atrous convolution.
    outputs_collections: Collection to add the ResNet unit output.
    scope: Optional variable_scope.

  Returns:
    The ResNet unit's output.
  """
  with tf.variable_scope(scope, 'bottleneck_v1', [inputs]) as sc:
    depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    if depth == depth_in:
      shortcut = subsample3D(inputs, stride, 'shortcut')
    else:
      shortcut = unit3D(inputs, depth, [1, 1, 1], strides=stride,
              activation_fn=None, name='shortcut')

    residual = unit3D(inputs, depth_bottleneck, [1, 1, 1], strides=1, name='conv1')
    residual = unit3D_same(residual, depth_bottleneck, 3, stride,
              rate=rate, name='conv2')
    residual = unit3D(residual, depth, [1, 1, 1], strides=1,
            activation_fn=None, name='conv3')

    output = tf.nn.relu(shortcut + residual)

    return slim.utils.collect_named_outputs(outputs_collections,
                                            sc.original_name_scope,
                                            output)

@add_arg_scope
def bottleneck3D_v2(inputs, depth, depth_bottleneck, stride, rate=1,
                    outputs_collections=None, scope=None):
  with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
    depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    input_shape = inputs.get_shape().as_list()
    preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')
    if depth == depth_in:
      shortcut = subsample3D(inputs, stride, 'shortcut')
    else:
      shortcut = unit3D(preact, depth, [1, 1, 1], strides=stride,
                                use_batch_norm=False, activation_fn=None,
                                name='shortcut')
    residual = unit3D(preact, depth_bottleneck, [1, 1, 1], strides=1,
                            name='conv1')
    residual = unit3D_same(residual, depth_bottleneck, 3, stride,
                                            rate=rate, name='conv2')
    residual = unit3D(residual, depth, [1, 1, 1], strides=1,
                            use_batch_norm=False, activation_fn=None,
                            name='conv3')
    output = shortcut + residual
    return slim.utils.collect_named_outputs(outputs_collections, 
                                            sc.original_name_scope,
                                            output)


@add_arg_scope
def stack_blocks_dense(net, blocks, output_stride=None,
                       outputs_collections=None):
  """Stacks ResNet `Blocks` and controls output feature density.

  First, this function creates scopes for the ResNet in the form of
  'block_name/unit_1', 'block_name/unit_2', etc.

  Second, this function allows the user to explicitly control the ResNet
  output_stride, which is the ratio of the input to output spatial resolution.
  This is useful for dense prediction tasks such as semantic segmentation or
  object detection.

  Most ResNets consist of 4 ResNet blocks and subsample the activations by a
  factor of 2 when transitioning between consecutive ResNet blocks. This results
  to a nominal ResNet output_stride equal to 8. If we set the output_stride to
  half the nominal network stride (e.g., output_stride=4), then we compute
  responses twice.

  Control of the output feature density is implemented by atrous convolution.

  Args:
    net: A `Tensor` of size [batch, height, width, channels].
    blocks: A list of length equal to the number of ResNet `Blocks`. Each
    element is a ResNet `Block` object describing the units in the `Block`.
    output_stride: If `None`, then the output will be computed at the nominal
    network stride. If output_stride is not `None`, it specifies the requested
    ratio of input to output spatial resolution, which needs to be equal to
    the product of unit strides from the start up to some level of the ResNet.
    For example, if the ResNet employs units with strides 1, 2, 1, 3, 4, 1,
    then valid values for the output_stride are 1, 2, 6, 24 or None (which
    is equivalent to output_stride=24).
    outputs_collections: Collection to add the ResNet block outputs.

  Returns:
    net: Output tensor with stride equal to the specified output_stride.

  Raises:
    ValueError: If the target output_stride is not valid.
  """
  # The current_stride variable keeps track of the effective stride of the
  # activations. This allows us to invoke atrous convolution whenever applying
  # the next residual unit would result in the activations having stride larger
  # than the target output_stride.
  current_stride = 1

  # The atrous convolution rate parameter.
  rate = 1

  for block in blocks:
    with tf.variable_scope(block.scope, 'block', [net]) as sc:
      for i, unit in enumerate(block.args):
        if output_stride is not None and current_stride > output_stride:
          raise ValueError('The target output_stride cannot be reached.')

        with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
          unit_depth, unit_depth_bottleneck, unit_stride = unit
          # If we have reached the target output_stride, then we need to employ
          # atrous convolution with stride=1 and multiply the atrous rate by the
          # current unit's stride for use in subsequent layers.
          if output_stride is not None and current_stride == output_stride:
            net = block.unit_fn(net, depth=unit_depth,
                      depth_bottleneck=unit_depth_bottleneck,
                      stride=1,
                      rate=rate)
            rate *= unit_stride

          else:
            net = block.unit_fn(net, depth=unit_depth,
                      depth_bottleneck=unit_depth_bottleneck,
                      stride=unit_stride,
                      rate=1)
            current_stride *= unit_stride
    net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

  if output_stride is not None and current_stride != output_stride:
    raise ValueError('The target output_stride cannot be reached.')

  return net