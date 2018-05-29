"""Inception-v2 Inflated 3D ConvNet architecture, the Inception-v2
network is introduced at TensorFlow slim library, see
https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v2.py,
also refer to the paper http://arxiv.org/abs/1502.03167.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from snets.net_utils import unit3D

def I3D_V2(inputs,
		   num_classes=101,
		   is_training=True,
		   final_endpoint='Prediction',
		   data_format='NHWC',
		   dropout_keep_prob=0.5,
		   min_depth=16,
		   depth_multiplier=1.0,
		   scope=None):
	end_points = {}
	if depth_multiplier <= 0:
		raise ValueError('depth_multiplier is not greater than zero.')
	depth = lambda d: max(int(d * depth_multiplier), min_depth)

	concat_axis = 2 if data_format == 'NCHW' else -1
	with tf.variable_scope(scope, 'I3D_V2', [inputs]):
		end_point = 'Conv3d_1a_7x7x7'
		net = unit3D(inputs, depth(64), [7,7,7], 2, is_training=is_training, name=end_point)
		end_points[end_point] = net
		if end_point == final_endpoint: return net, end_points
		end_point = 'MaxPool3d_2a_1x3x3'
		net = tf.nn.max_pool3d(net, [1, 1, 3, 3, 1], [1, 1, 2, 2, 1], padding='SAME', name=end_point)
		end_points[end_point] = net
		if end_point == final_endpoint: return net, end_points
		end_point = 'Conv3d_2b_1x1x1'
		net = unit3D(net, depth(64), [1, 1, 1], is_training=is_training, name=end_point)
		end_points[end_point] = net
		if end_point == final_endpoint: return net, end_points
		end_point = 'Conv3d_2c_3x3x3'
		net = unit3D(net, depth(192), [3, 3, 3], is_training=is_training, name=end_point)
		end_points[end_point] = net
		if end_point == final_endpoint: return net, end_points
		end_point = 'MaxPool3d_3a_1x3x3'
		net = tf.nn.max_pool3d(net, [1, 1, 3, 3, 1], [1, 1, 2, 2, 1], padding='SAME', name=end_point)
		end_points[end_point] = net
		if end_point == final_endpoint: return net, end_points
		end_point = 'Mixed_3b'
		with tf.variable_scope(end_point):
			with tf.variable_scope('Branch_0'):
				branch_0 = unit3D(net, depth(64), kernel_shape=[1, 1, 1],
						    is_training=is_training, name='Conv3d_0a_1x1x1')
			with tf.variable_scope('Branch_1'):
				branch_1 = unit3D(net, depth(64), kernel_shape=[1, 1, 1],
						    is_training=is_training, name='Conv3d_0a_1x1x1')
				branch_1 = unit3D(branch_1, depth(64), kernel_shape=[3, 3, 3],
						    is_training=is_training, name='Conv3d_0b_3x3x3')
			with tf.variable_scope('Branch_2'):
				branch_2 = unit3D(net, depth(64), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0a_1x1x1')
				branch_2 = unit3D(branch_2, depth(96), kernel_shape=[3, 3, 3],
								  is_training=is_training, name='Conv3d_0b_3x3x3')
				branch_2 = unit3D(branch_2, depth(96), kernel_shape=[3, 3, 3],
								  is_training=is_training, name='Conv3d_0c_3x3x3')
			with tf.variable_scope('Branch_3'):
				branch_3 = tf.nn.max_pool3d(net, [1, 3, 3, 3, 1], strides=[1, 1, 1, 1, 1],
											padding='SAME', name='MaxPool3d_0a_3x3x3')
				branch_3 = unit3D(branch_3, depth(32), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0b_1x1x1')
			net = tf.concat([branch_0, branch_1, branch_2, branch_3], concat_axis)

		end_points[end_point] = net
		if end_point == final_endpoint: return net, end_points
		end_point = 'Mixed_3c'
		with tf.variable_scope(end_point):
			with tf.variable_scope('Branch_0'):
				branch_0 = unit3D(net, depth(64), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0a_1x1x1')
			with tf.variable_scope('Branch_1'):
				branch_1 = unit3D(net, depth(64), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0a_1x1x1')
				branch_1 = unit3D(branch_1, depth(96), kernel_shape=[3, 3, 3],
								  is_training=is_training, name='Conv3d_0b_3x3x3')
			with tf.variable_scope('Branch_2'):
				branch_2 = unit3D(net, depth(64), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0a_1x1x1')
				branch_2 = unit3D(branch_2, depth(96), kernel_shape=[3, 3, 3],
								  is_training=is_training, name='Conv3d_0b_3x3x3')
				branch_2 = unit3D(branch_2, depth(96), kernel_shape=[3, 3, 3],
								  is_training=is_training, name='Conv3d_0c_3x3x3')
			with tf.variable_scope('Branch_3'):
				branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
											strides=[1, 1, 1, 1, 1], padding='SAME',
											name='MaxPool3d_0a_3x3x3')
				branch_3 = unit3D(branch_3, depth(64), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0b_1x1x1')
		net = tf.concat([branch_0, branch_1, branch_2, branch_3], concat_axis)
		end_points[end_point] = net
		if end_point == final_endpoint: return net, end_points

		end_point = 'Mixed_4a'
		with tf.variable_scope(end_point):
			with tf.variable_scope('Branch_0'):
				branch_0 = unit3D(net, depth(128), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0a_1x1x1')
				branch_0 = unit3D(branch_0, depth(160), kernel_shape=[3, 3, 3],
                            	  strides=[2, 2, 2], is_training=is_training, name='Conv3d_1a_3x3x3')
			with tf.variable_scope('Branch_1'):
				branch_1 = unit3D(net, depth(64), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0a_1x1x1')
				branch_1 = unit3D(branch_1, depth(64), kernel_shape=[3, 3, 3],
				            	  is_training=is_training, name='Conv3d_0b_3x3x3')
				branch_1 = unit3D(branch_1, depth(64), kernel_shape=[3, 3, 3],
				            	  strides=[2, 2, 2], is_training=is_training, name='Conv3d_1a_3x3x3')
			with tf.variable_scope('Branch_2'):
				branch_2 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
											strides=[1, 2, 2, 2, 1], padding='SAME',
                                      		name='MaxPool3d_1a_3x3x3')
			net = tf.concat([branch_0, branch_1, branch_2], concat_axis)
		end_points[end_point] = net
		if end_point == final_endpoint: return net, end_points
		end_point = 'Mixed_4b'
		with tf.variable_scope(end_point):
			with tf.variable_scope('Branch_0'):
				branch_0 = unit3D(net, depth(224), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv2d_0a_1x1x1')
			with tf.variable_scope('Branch_1'):
				branch_1 = unit3D(net, depth(64), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv2d_0a_1x1x1')
				branch_1 = unit3D(branch_1, depth(96), kernel_shape=[3, 3, 3],
								  is_training=is_training, name='Conv2d_0b_3x3x3')
			with tf.variable_scope('Branch_2'):
				branch_2 = unit3D(net, depth(96), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv2d_0a_1x1x1')
				branch_2 = unit3D(branch_2, depth(128), kernel_shape=[3, 3, 3],
								  is_training=is_training, name='Conv2d_0b_3x3x3')
				branch_2 = unit3D(branch_2, depth(128), kernel_shape=[3, 3, 3],
								  is_training=is_training, name='Conv2d_0c_3x3x3')
			with tf.variable_scope('Branch_3'):
				branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
											strides=[1, 1, 1, 1, 1], padding='SAME',
                                      		name='MaxPool3d_0a_3x3x3')
				branch_3 = unit3D(branch_3, depth(128), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv2d_0b_1x1x1')
			net = tf.concat(axis=concat_axis, values=[branch_0, branch_1, branch_2, branch_3])
		end_points[end_point] = net
		if end_point == final_endpoint: return net, end_points
		end_point = 'Mixed_4c'
		with tf.variable_scope(end_point):
			with tf.variable_scope('Branch_0'):
				branch_0 = unit3D(net, depth(192), [1, 1, 1],
								  is_training=is_training, name='Conv2d_0a_1x1x1')
			with tf.variable_scope('Branch_1'):
				branch_1 = unit3D(net, depth(96), [1, 1, 1],
								  is_training=is_training, name='Conv2d_0a_1x1x1')
				branch_1 = unit3D(branch_1, depth(128), [3, 3, 3],
								  is_training=is_training, name='Conv2d_0b_3x3x3')
			with tf.variable_scope('Branch_2'):
				branch_2 = unit3D(net, depth(96), [1, 1, 1],
								  is_training=is_training, name='Conv2d_0a_1x1x1')
				branch_2 = unit3D(branch_2, depth(128), [3, 3, 3],
								  is_training=is_training, name='Conv2d_0b_3x3x3')
				branch_2 = unit3D(branch_2, depth(128), [3, 3, 3],
								  is_training=is_training, name='Conv2d_0c_3x3x3')
			with tf.variable_scope('Branch_3'):
				branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                      		strides=[1, 1, 1, 1, 1], padding='SAME',
                                      		name='MaxPool3d_0a_3x3x3')
				branch_3 = unit3D(branch_3, depth(128), [1, 1, 1],
								  is_training=is_training, name='Conv2d_0b_1x1x1')
			net = tf.concat(axis=concat_axis, values=[branch_0, branch_1, branch_2, branch_3])
		end_points[end_point] = net
		if end_point == final_endpoint: return net, end_points

		# 14 x 14 x 576
		end_point = 'Mixed_4d'
		with tf.variable_scope(end_point):
			with tf.variable_scope('Branch_0'):
				branch_0 = unit3D(net, depth(160), [1, 1, 1],
								  is_training=is_training, name='Conv2d_0a_1x1x1')
			with tf.variable_scope('Branch_1'):
				branch_1 = unit3D(net, depth(128), [1, 1, 1],
								  is_training=is_training, name='Conv2d_0a_1x1x1')
				branch_1 = unit3D(branch_1, depth(160), [3, 3, 3],
								  is_training=is_training, name='Conv2d_0b_3x3x3')
			with tf.variable_scope('Branch_2'):
				branch_2 = unit3D(net, depth(128), [1, 1, 1],
								  is_training=is_training, name='Conv2d_0a_1x1x1')
				branch_2 = unit3D(branch_2, depth(160), [3, 3, 3],
								  is_training=is_training, name='Conv2d_0b_3x3x3')
				branch_2 = unit3D(branch_2, depth(160), [3, 3, 3],
								  is_training=is_training, name='Conv2d_0c_3x3x3')
			with tf.variable_scope('Branch_3'):
				branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                      		strides=[1, 1, 1, 1, 1], padding='SAME',
                                      		name='MaxPool3d_0a_3x3x3')
				branch_3 = unit3D(branch_3, depth(96), [1, 1, 1],
								  is_training=is_training, name='Conv2d_0b_1x1x1')
			net = tf.concat(axis=concat_axis, values=[branch_0, branch_1, branch_2, branch_3])
		end_points[end_point] = net
		if end_point == final_endpoint: return net, end_points
		# 14 x 14 x 576
		end_point = 'Mixed_4e'
		with tf.variable_scope(end_point):
			with tf.variable_scope('Branch_0'):
				branch_0 = unit3D(net, depth(96), [1, 1, 1],
								  is_training=is_training, name='Conv2d_0a_1x1x1')
			with tf.variable_scope('Branch_1'):
				branch_1 = unit3D(net, depth(128), [1, 1, 1],
								  is_training=is_training, name='Conv2d_0a_1x1x1')
				branch_1 = unit3D(branch_1, depth(192), [3, 3, 3],
								  is_training=is_training, name='Conv2d_0b_3x3x3')
			with tf.variable_scope('Branch_2'):
				branch_2 = unit3D(net, depth(160), [1, 1, 1],
								  is_training=is_training, name='Conv2d_0a_1x1x1')
				branch_2 = unit3D(branch_2, depth(192), [3, 3, 3],
								  is_training=is_training, name='Conv2d_0b_3x3x3')
				branch_2 = unit3D(branch_2, depth(192), [3, 3, 3],
								  is_training=is_training, name='Conv2d_0c_3x3x3')
			with tf.variable_scope('Branch_3'):
				branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                      		strides=[1, 1, 1, 1, 1], padding='SAME',
                                      		name='MaxPool3d_0a_3x3x3')
				branch_3 = unit3D(branch_3, depth(96), [1, 1, 1],
								  is_training=is_training, name='Conv2d_0b_1x1x1')
			net = tf.concat(axis=concat_axis, values=[branch_0, branch_1, branch_2, branch_3])
		end_points[end_point] = net
		if end_point == final_endpoint: return net, end_points

		end_point = 'Mixed_5a'
		with tf.variable_scope(end_point):
			with tf.variable_scope('Branch_0'):
				branch_0 = unit3D(net, depth(128), [1, 1, 1],
								  is_training=is_training, name='Conv2d_0a_1x1x1')
				branch_0 = unit3D(branch_0, depth(192), [3, 3, 3], strides=2,
								  is_training=is_training, name='Conv2d_1a_3x3x3')
			with tf.variable_scope('Branch_1'):
				branch_1 = unit3D(net, depth(192), [1, 1, 1],
								  is_training=is_training, name='Conv2d_0a_1x1x1')
				branch_1 = unit3D(branch_1, depth(256), [3, 3, 3],
								  is_training=is_training, name='Conv2d_0b_3x3x3')
				branch_1 = unit3D(branch_1, depth(256), [3, 3, 3], strides=2,
								  is_training=is_training, name='Conv2d_1a_3x3x3')
			with tf.variable_scope('Branch_2'):
				branch_2 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
											strides=[1, 2, 2, 2, 1], padding='SAME',
                                      		name='MaxPool3d_1a_3x3x3')
			net = tf.concat(axis=concat_axis, values=[branch_0, branch_1, branch_2])
		end_points[end_point] = net
		if end_point == final_endpoint: return net, end_points
		# 7 x 7 x 1024
		end_point = 'Mixed_5b'
		with tf.variable_scope(end_point):
			with tf.variable_scope('Branch_0'):
				branch_0 = unit3D(net, depth(352), [1, 1, 1],
								  is_training=is_training, name='Conv2d_0a_1x1x1')
			with tf.variable_scope('Branch_1'):
				branch_1 = unit3D(net, depth(192), [1, 1, 1],
								  is_training=is_training, name='Conv2d_0a_1x1x1')
				branch_1 = unit3D(branch_1, depth(320), [3, 3, 3],
								  is_training=is_training, name='Conv2d_0b_3x3x3')
			with tf.variable_scope('Branch_2'):
				branch_2 = unit3D(net, depth(160), [1, 1, 1],
								  is_training=is_training, name='Conv2d_0a_1x1x1')
				branch_2 = unit3D(branch_2, depth(224), [3, 3, 3],
								  is_training=is_training, name='Conv2d_0b_3x3x3')
				branch_2 = unit3D(branch_2, depth(224), [3, 3, 3],
								  is_training=is_training, name='Conv2d_0c_3x3x3')
			with tf.variable_scope('Branch_3'):
				branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                      		strides=[1, 1, 1, 1, 1], padding='SAME',
                                      		name='MaxPool3d_0a_3x3x3')
				branch_3 = unit3D(branch_3, depth(128), [1, 1, 1],
								  is_training=is_training, name='Conv2d_0b_1x1x1')
			net = tf.concat(axis=concat_axis, values=[branch_0, branch_1, branch_2, branch_3])
		end_points[end_point] = net
		if end_point == final_endpoint: return net, end_points

		end_point = 'Mixed_5c'
		with tf.variable_scope(end_point):
			with tf.variable_scope('Branch_0'):
				branch_0 = unit3D(net, depth(352), [1, 1, 1],
								  is_training=is_training, name='Conv2d_0a_1x1x1')
			with tf.variable_scope('Branch_1'):
				branch_1 = unit3D(net, depth(192), [1, 1, 1],
								  is_training=is_training, name='Conv2d_0a_1x1x1')
				branch_1 = unit3D(branch_1, depth(320), [3, 3, 3],
								  is_training=is_training, name='Conv2d_0b_3x3x3')
			with tf.variable_scope('Branch_2'):
				branch_2 = unit3D(net, depth(192), [1, 1, 1],
								  is_training=is_training, name='Conv2d_0a_1x1x1')
				branch_2 = unit3D(branch_2, depth(224), [3, 3, 3],
								  is_training=is_training, name='Conv2d_0b_3x3x3')
				branch_2 = unit3D(branch_2, depth(224), [3, 3, 3],
								  is_training=is_training, name='Conv2d_0c_3x3x3')
			with tf.variable_scope('Branch_3'):
				branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                      		strides=[1, 1, 1, 1, 1], padding='SAME',
                                      		name='MaxPool3d_0a_3x3x3')
				branch_3 = unit3D(branch_3, depth(128), [1, 1, 1],
								  is_training=is_training, name='Conv2d_0b_1x1x1')
			net = tf.concat(axis=concat_axis, values=[branch_0, branch_1, branch_2, branch_3])
		end_points[end_point] = net
		if end_point == final_endpoint: return net, end_points

		end_point = 'Logits'
		with tf.variable_scope(end_point):
			net = tf.nn.avg_pool3d(net, ksize=[1, 2, 7, 7, 1],
								   strides=[1, 1, 1, 1, 1], padding='VALID')
			print(net)
			net = tf.nn.dropout(net, dropout_keep_prob)
			logits = unit3D(net, num_classes,
							kernel_shape=[1, 1, 1],
							activation_fn=None,
							is_training=is_training,
							use_batch_norm=False,
							use_bias=True,
							name='Conv3d_1c_1x1x1')
			logits = tf.squeeze(logits, [2, 3], name='SpatialSqueeze')
			averaged_logits = tf.reduce_mean(logits, axis=1)
		end_points[end_point] = averaged_logits
		if end_point == final_endpoint: return logits, end_points
		end_point = 'Predictions'
		predictions = tf.nn.softmax(averaged_logits)
		end_points[end_point] = predictions
		if end_point == final_endpoint: return predictions, end_points

if __name__ == '__main__':
	# inputs: [batch_size, num_frames, h, w, c], outputs: [batch_size, num_classes]
	inps = tf.placeholder(dtype=tf.float32, shape=[4, 64, 224, 224, 3])
	si3d, _ = I3D_V2(inps, final_endpoint='Predictions')
	print(si3d)
