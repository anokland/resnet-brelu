# resnet-brelu
ResNet and DenseNet with Bipolar Rectified Linear Units

This repo is copied from:
https://github.com/facebook/fb.resnet.torch

Then, the very deep 1K layer model (resnet-pre-act.lua) is copied from here:
https://github.com/KaimingHe/resnet-1k-layers

Then, the DenseNet model (densenet.lua) is copied from here:
https://github.com/liuzhuang13/DenseNet

Then, the code is modified in the following ways:
- Added command line option for replacing ReLU with BReLU
- Added command line options for DenseNet
- Use modelType command line option to decide learning rate decay for DenseNet
- Added logging of training and test error to log.txt every epoch
- Added logging of command line options to file opt.txt
- Added logging of model as text to model.txt
- Commented out debug printout every batch

The current implementation of BReLU does not support training on multiple GPU's.

Results with BReLU compared to ReLU on Cifar-10 with data augmentation:

| Network         | ReLU           | BReLU 1   | BReLU 2 | BReLU 3 | BReLU 4 | BReLU 5 | BreLU        |
| :---            | :---           | :---      | :---    | :---    | :---    | :---    | :---         |
| ResNet-110      | 6.61 / 6.41    | 5.724     |         |         |         |         | 5.725 +- 0.0 |
| DenseNet-BC-100 | 4.51           |           |         |         |         |         |              |
| DenseNet-BC-190 | 3.46 (SOTA)    |           |         |         |         |         |              |

Results with BReLU compared to ReLU on Cifar-100 with data augmentation:

| Network         | ReLU           | BReLU 1   | BReLU 2 | BReLU 3 | BReLU 4 | BReLU 5 | BreLU        |
| :---            | :---           | :---      | :---    | :---    | :---    | :---    | :---         |
| ResNet-110      | 27.22          |           |         |         |         |         | 0.0 +- 0.0   |
| DenseNet-BC-100 | 22.27          |           |         |         |         |         |              |
| DenseNet-BC-190 | 17.18 (SOTA)   |           |         |         |         |         |              |


Training recipes
----------------

### CIFAR-10

To train ResNet-20 on CIFAR-10 with 2 GPUs:

```bash
th main.lua -dataset cifar10 -nGPU 2 -batchSize 128 -depth 20
```

To train ResNet-110 instead just change the `-depth` flag:

```bash
th main.lua -dataset cifar10 -nGPU 2 -batchSize 128 -depth 110
```

To train a pre-activation ResNet-1001 with BReLU with 1 GPU:

```bash
th main.lua -netType resnet-pre-act -depth 1001 -batchSize 64 -nGPU 1 -dataset cifar10 -nEpochs 200 -BReLU true
```
To train a ResNet110 with BReLU with 1 GPU:

```bash
th main.lua -netType resnet -depth 110 -batchSize 128 -nGPU 1 -dataset cifar10 -BReLU true
```

To train a DenseNet-100 with BReLU with 1 GPU:

```bash
th main.lua -netType densenet -depth 100 -growthRate 12 -nGPU 1 -dataset cifar10 -batchSize 64 -nEpochs 300 -optnet true -BReLU true
```

To train a DenseNet-190 with BReLU with 1 GPU:

```bash
th main.lua -netType densenet -depth 190 -growthRate 40 -nGPU 1 -dataset cifar10 -batchSize 64 -nEpochs 300 -optnet true -BReLU true
```
