"""Convert weights from kinetics and imagenet pretrained model
rebuild_ckpoint_kinetics: reconstruct an I3D model based on the pure 
tensorflow framework, where weights are initialized from the kinetics-i3d model
"""

import os
import sys
import argparse
import tensorflow as tf

KINETICS_NAME_MAP = {
    'RGB/inception_i3d/': 'v/SenseTime_I3D/',
    'batch_norm': 'BatchNorm',
    'conv_3d/w': 'conv3d/kernel',
    'conv_3d/b': 'conv3d/bias',
    '1x1': '1x1x1',
    '3x3': '3x3x3',
    '7x7': '7x7x7'
}

def rebuild_ckpoint_kinetics(checkpoint_dir, save_path):
    """rebuild the checkpoint from kinetics-i3d model """
    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
    with tf.Session() as sess:
        for var_name, _ in tf.contrib.framework.list_variables(checkpoint_dir):
            raw_var = tf.contrib.framework.load_variable(checkpoint_dir, var_name)
            for k, v in KINETICS_NAME_MAP.items():
                var_name = var_name.replace(k, v)
            print(var_name, raw_var.shape)
            var = tf.Variable(raw_var, name=var_name)

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    saver.save(sess, save_path)

def rebuild_ckpoint_imagenet(checkpoint_dir, save_path):
    """rebuild the checkpoint from kinetics-i3d model """
    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
    with tf.Session() as sess:
        for var_name, _ in tf.contrib.framework.list_variables(checkpoint_dir):
            raw_var = tf.contrib.framework.load_variable(checkpoint_dir, var_name)
            for k, v in KINETICS_NAME_MAP.items():
                var_name = var_name.replace(k, v)
            print(var_name, raw_var.shape)
            var = tf.Variable(raw_var, name=var_name)

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    saver.save(sess, save_path)

def main():
    checkpoint_dir = './kinetics-i3d/data/checkpoints/rgb_imagenet/model.ckpt'
    rebuild_ckpoint_kinetics(checkpoint_dir, './kinetics-i3d/data/kinetics_i3d/model')

if __name__ == '__main__':
    main()