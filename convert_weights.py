import os
import numpy as np
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

IMAGENET_NAME_MAP = {
    'InceptionV2/': 'v/SenseTime_I3D_V2/',
    'Conv2d': 'Conv3d',
    'weights': 'kernel',
    '1x1': '1x1x1',
    '3x3': '3x3x3',
    '7x7': '7x7x7'
}

def rebuild_ckpoint_kinetics(checkpoint_dir, save_path):
    var_list = {} # To store the variables
    for var_name, _ in tf.train.list_variables(checkpoint_dir):
        raw_var = tf.train.load_variable(checkpoint_dir, var_name)
        for k, v in KINETICS_NAME_MAP.items():
            var_name = var_name.replace(k, v)
        print(var_name, raw_var.shape)
        var_list[var_name] = tf.Variable(raw_var, name=var_name)

    # Save new variables
    ckpt = tf.train.Checkpoint(**var_list)
    ckpt.save(save_path)

def rebuild_ckpoint_imagenet(checkpoint_dir, save_path):
    var_list = {} # To store the variables
    fg = True
    for var_name, _ in tf.train.list_variables(checkpoint_dir):
        raw_var = tf.train.load_variable(checkpoint_dir, var_name)
        if var_name.startswith('InceptionV2/Conv2d_1a_7x7'):
            if fg:
                var_name = 'v/SenseTime_I3D_V2/Conv3d_1a_7x7x7/kernel'
                raw_var = np.random.normal(0.0, 1.0, (7, 7, 7, 3, 64)) / 7.0
                fg = False
            else:
                continue
        elif var_name.find('weights') > -1:
            kernel = raw_var.shape[0]
            res = [raw_var for i in range(kernel)]
            raw_var = np.stack(res, axis=0)
            raw_var = raw_var / (kernel * 1.0)
        for k, v in IMAGENET_NAME_MAP.items():
            var_name = var_name.replace(k, v)
        print(var_name, raw_var.shape)
        var_list[var_name] = tf.Variable(raw_var, name=var_name)

    # Save new variables
    ckpt = tf.train.Checkpoint(**var_list)
    ckpt.save(save_path)

def main():
    checkpoint_dir = './kinetics-i3d/data/checkpoints/rgb_imagenet/model.ckpt'
    rebuild_ckpoint_kinetics(checkpoint_dir, './kinetics-i3d/data/kinetics_i3d/model')
    # checkpoint_dir = './kinetics-i3d/data/2dmodel/inception_v2.ckpt'
    # rebuild_ckpoint_imagenet(checkpoint_dir, './kinetics-i3d/data/inceptionv2_i3d/model')

if __name__ == '__main__':
    main()
