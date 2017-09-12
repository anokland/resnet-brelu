# resnet-brelu
ResNet and DenseNet with Bipolar Rectified Linear Units

This repo is copied from:
https://github.com/facebook/fb.resnet.torch

Then, the very deep 1K layer model (resnet-pre-act.lua) is copied from here:
https://github.com/KaimingHe/resnet-1k-layers

Then, the DenseNet model (densenet.lua) is copied from here:
https://github.com/liuzhuang13/DenseNet

Then, the code is modified in the following ways:
- Added command line option for replacing ReLU with BReLU (`BReLU`)
- Added command line options for DenseNet (`growthRate`, `dropRate` and `bottleneck`)
- Train Cifar-10 and Cifar-100 networks for 300 epochs
- Train Cifar-10 and Cifar-100 networks with learning rate decay at epoch 150 and 225 (as prescribed for densenet)
- Added logging of training and test error to log.txt every epoch
- Added logging of command line options to file opt.txt at startup
- Added logging of model structure as text to model.txt at startup
- Commented out debug printout every batch of training and validation

The current implementation of BReLU does not support training on multiple GPU's.

Results with BReLU compared to ReLU on Cifar-10 with data augmentation:

| Network         | ReLU (own)       | BReLU (run1)   | BReLU (run2) | BReLU (run3) | BReLU (run4) |
| :---            | :---             | :---           | :---         | :---         | :---         |
| ResNet-44       | 7.17 (6.44)      | 6.49           | 6.80         | 6.64         | 6.61         | 
| ResNet-110      | 6.61/6.41 (6.92) | 5.76           | 6.10         | 5.86         | 6.07         | 

Results with BReLU compared to ReLU on Cifar-100 with data augmentation:

| Network         | ReLU (own)     | BReLU (run1)   | BReLU (run2) | BReLU (run3) | BReLU (run4) |
| :---            | :---           | :---           | :---         | :---         | :---         | 
| ResNet-44       | 27.76 (29.4)   | 29.30          | 29.96        | 29.85        | 29.67        |
| ResNet-110      | 27.22          | 27.74          | 28.04        | 28.41        | 28.11        | 


Training recipes
----------------

### CIFAR-10

To train a pre-activation ResNet-1001 with BReLU with 1 GPU:

```bash
th main.lua -netType resnet-pre-act -depth 1001 -batchSize 64 -nGPU 1 -dataset cifar10 -nEpochs 200 -BReLU true
```

To train a ResNet-44 with BReLU with 1 GPU:

```bash
th main.lua -netType resnet -depth 44 -batchSize 128 -nGPU 1 -dataset cifar10 -BReLU true
```

To train a ResNet-110 with BReLU with 1 GPU:

```bash
th main.lua -netType resnet -depth 110 -batchSize 128 -nGPU 1 -dataset cifar10 -BReLU true
```

To train a DenseNet-100 with BReLU with 1 GPU:

```bash
th main.lua -netType densenet -depth 100 -batchSize 128 -growthRate 12 -nGPU 1 -dataset cifar10 -optnet true -BReLU true
```

To train a DenseNet-190 with BReLU with 1 GPU:

```bash
th main.lua -netType densenet -depth 190 -batchSize 64 -growthRate 40 -nGPU 1 -dataset cifar10 -optnet true -BReLU true
```
