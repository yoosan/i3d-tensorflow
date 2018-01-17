# I3D-TensorFlow

This repo contains the inflated version of the recently popular ConvNets.

## Convert weights from the pretrained model
DeepMind have provided the Inception-v1 inflated 3D model, building upon the [sonnet](https://github.com/deepmind/sonnet). I slightly modified their code and rewrited the i3d model using the protogenetic tensorflow op. The pretrained weights from kinetics-i3d can be easily migrated to the new model.
To botain the weights from kinetics-i3d, execute the following instructions.
```bash
$ git clone https://github.com/yoosan/i3d-tensorflow
$ cd i3d-tensorflow
$ git clone https://github.com/deepmind/kinetics-i3d
$ python convert_weights.py
```

## Training the I3D model on UCF101
Now I'm preparing the code for training model on the UCF101 dataset. Using the kinetics pretrained weights, we achive a result of **95.2%** top-1 accuracy on split 1 of UCF101 with RGB modality.
