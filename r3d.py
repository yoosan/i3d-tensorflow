"""Inflated 3D resnet, including resnet50, resnet101, resnet_152
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import collections
from snets.scopes import *
import snets.net_utils as net_utils
slim = tf.contrib.slim
import numpy as np

def resnet_v1(inputs,
			  blocks,
			  num_classes=None,
			  is_training=True,
			  output_stride=None,
			  include_root_block=True,
			  dropout_keep_prob=0.5,
			  reuse=None,
			  scope=None):
	with tf.variable_scope(scope, 'resnet_v1', [inputs], reuse=reuse) as sc:
		end_points_collection = sc.name + '_end_points'
		with arg_scope([net_utils.unit3D, net_utils.bottleneck3D,
						net_utils.stack_blocks_dense]):
			net = inputs
			if include_root_block:
				if output_stride is not None:
					if output_stride % 4 != 0:
						raise ValueError('The output_stride needs to be a multiple of 4.')
					output_stride /= 4
				net = net_utils.unit3D_same(net, 64, 7, strides=2, name='conv1')
				net = tf.nn.max_pool3d(net, [1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool1')
			net = net_utils.stack_blocks_dense(net, blocks, output_stride)
			net = tf.reduce_mean(net, [1, 2, 3], name='pool5', keep_dims=True)
			end_points = slim.utils.convert_collection_to_dict(end_points_collection)
			if num_classes is not None:
				net = slim.flatten(net)
				net = tf.nn.dropout(net, keep_prob=dropout_keep_prob)
				net = slim.fully_connected(net, num_classes, activation_fn=None, scope='logits')
				net = tf.nn.softmax(net, name='predictions')
				end_points['predictions'] = net
			return net, end_points

resnet_v1.default_image_size = 224

def resnet_v1_50(inputs,
				 num_classes=101,
				 is_training=True,
				 data_format='NCHW',
				 global_pool=True,
				 dropout_keep_prob=0.5,
				 output_stride=None,
         final_endpoint='Prediction',
				 reuse=None,
				 scope='resnet_v1_50'):
	"""ResNet-50 model of [1]. See resnet_v1() for arg and return description."""
	blocks = [
		net_utils.Block(
			'block1', net_utils.bottleneck3D, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
		net_utils.Block(
			'block2', net_utils.bottleneck3D, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
		net_utils.Block(
			'block3', net_utils.bottleneck3D, [(1024, 256, 1)] * 5 + [(1024, 256, 2)]),
		net_utils.Block(
			'block4', net_utils.bottleneck3D, [(2048, 512, 1)] * 3),
	]
	net = resnet_v1(inputs, blocks, num_classes, is_training=True,
					output_stride=output_stride,
					dropout_keep_prob=dropout_keep_prob,
					include_root_block=True, reuse=reuse, scope=scope)
	return net
resnet_v1_50.default_image_size = resnet_v1.default_image_size


"""
blocks = [
      resnet_utils.Block(
          'block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
      resnet_utils.Block(
          'block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
      resnet_utils.Block(
          'block3', bottleneck, [(1024, 256, 1)] * 22 + [(1024, 256, 2)]),
      resnet_utils.Block(
          'block4', bottleneck, [(2048, 512, 1)] * 3)
  ]
"""

def resnet_v1_101(inputs,
				  num_classes,
				  is_training=True,
				  data_format='NCHW',
				  global_pool=True,
				  dropout_keep_prob=0.5,
				  output_stride=None,
				  reuse=None,
				scope='resnet_v1_101'):
	"""ResNet-101 model of [1]. See resnet_v1() for arg and return description."""
	blocks = [
		net_utils.Block(
			'block1', net_utils.bottleneck3D, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
		net_utils.Block(
			'block2', net_utils.bottleneck3D, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
		net_utils.Block(
			'block3', net_utils.bottleneck3D, [(1024, 256, 1)] * 22 + [(1024, 256, 2)]),
		net_utils.Block(
			'block4', net_utils.bottleneck3D, [(2048, 512, 1)] * 3),
	]
	net = resnet_v1(inputs, blocks, num_classes, is_training=True,
					output_stride=output_stride,
					include_root_block=True, reuse=reuse, scope=scope)
	return net
resnet_v1_101.default_image_size = resnet_v1.default_image_size


def resnet_v1_152(inputs,
				  num_classes,
				  is_training=True,
				  data_format='NCHW',
				  global_pool=True,
				  dropout_keep_prob=0.5,
				  output_stride=None,
				  reuse=None,
				  scope='resnet_v1_101'):
	"""ResNet-101 model of [1]. See resnet_v1() for arg and return description."""
	blocks = [
		net_utils.Block(
			'block1', net_utils.bottleneck3D, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
		net_utils.Block(
			'block2', net_utils.bottleneck3D, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
		net_utils.Block(
			'block3', net_utils.bottleneck3D, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
		net_utils.Block(
			'block4', net_utils.bottleneck3D, [(2048, 512, 1)] * 3),
	]
	net = resnet_v1(inputs, blocks, num_classes, is_training=True,
					output_stride=output_stride,
					dropout_keep_prob=dropout_keep_prob,
					include_root_block=True, reuse=reuse, scope=scope)
	return net
resnet_v1_152.default_image_size = resnet_v1.default_image_size

if __name__ == '__main__':
	print('hello world')
	inps = tf.placeholder(dtype=tf.float32, shape=[4, 64, 224, 224, 3])
	test = net_utils.unit3D_same(inps, 64, 7, 2)
	print(test)
	res = net_utils.unit3D(inps, 64, 7, 2)
	print(res)

	bok =  net_utils.bottleneck3D(res, 128, 128, 1)
	print(bok)

	net = net_utils.unit3D_same(inps, 64, 7, strides=2, name='conv1')
	net = tf.nn.max_pool3d(net, [1, 3, 3, 3, 1], strides=[1,2,2,2,1], padding='SAME', name='pool1')
	print(net)
	blocks = [
		net_utils.Block(
			'block1', net_utils.bottleneck3D, [(256, 64, 1)] * 2 + [(256, 64, 2)])
	]
	net = net_utils.stack_blocks_dense(net, blocks, output_stride=None)
	print(net)
	res50 = resnet_v1_50(inps, num_classes=101)
	print(res50)
