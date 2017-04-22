# resnet-brelu
ResNet and DenseNet with Bipolar Rectified Linear Units

This repo is copied from:
https://github.com/facebook/fb.resnet.torch

Then, the very deep 1K layer model (resnet-pre-act.lua) is copied from here:
https://github.com/KaimingHe/resnet-1k-layers

Then, the DenseNet model (densenet.lua) is copied from here:
https://github.com/liuzhuang13/DenseNet

The code is modified in the following ways:
- Added command line option for replacing ReLU with BReLU
- Added command line options for DenseNet
- Use modelType command line option to decide learning rate decay for DenseNet
- Added logging of training and test error to log.txt every epoch
- Added logging of command line options to file opt.txt
- Added logging of model as text to model.txt
- Commented out debug printout every batch
