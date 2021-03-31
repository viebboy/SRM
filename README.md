# Knowledge Distillation by Sparse Representation Matching
This repository provides the implementation of the paper "Knowledge Distillation By Sparse Representation Matching".

## Dependency
The code is written in python 3.x with the following packages:
- mxnet 1.6.0 (required only for ImageNet experiments)
- gluoncv 0.6.0 (required only for ImageNet experiments)
- pytorch 1.4.0
- torchvision 0.5.0
- tqdm 4.43.0

## Data
The data with its train/val/test split should be downloaded from [this](https://drive.google.com/file/d/1lg6VncaTMMqPHf4DOgW576gHGGLU8oO3/view?usp=sharing) and should be put under the main directory. The directory structure should be like this:

``` 
SRM/
│   README.md
└───data/ 
    │   imagenet_train.rec
    │   imagenet_train.idx
    │   imagenet_val.rec
    │   imagenet_val.idx
    └───cifar100 
    └───cub
    │   ...
└───cifar100/
    └───code/
        │    AllCNN.py
        │    Datasets.py
        │    ...
└───transfer_learning/
    └───code/ 
        │    AllCNN.py
        │    Datasets.py
        │    ...
└───imagenet/
    └───code/
        │    Datasets.py
        │    exp_configurations.py
        │    ...

```

For ImageNet experiments, we used a preprocessing pipeline from mxnet, hence the required dependency. Please follow the instruction from [mxnet](https://gluon-cv.mxnet.io/build/examples_datasets/recordio.html) and prepare the ImageRecordIter files and put them under the data directory with the following names:

``` 
SRM/
│   README.md
└───data/
    │   imagenet_train.rec (ImageRecordIter file)
    │   imagenet_train.idx (index file) 
    │   imagenet_val.rec (ImageRecordIter file)
    │   imagenet_val.idx (index file)
    │   ...
```

## Code organization & usage 

Source code for cifar100, transfer learning and imagenet experiments is in the respective directory. The code has been structured according to the following modular format:
- Datasets.py handles the loading and preprocessing of data
- exp_configurations.py contains the specification of all hyperparameters related to each method
- Runners.py contains the main training + evaluation functions of each method
- train.py is a meta script to run all configurations of all methods 
- Utility.py contains miscellaneous functions that are shared by all methods 

For every method, the result of each configuration is saved in a pickle file under **output** directory. During training, the intermediate result is saved under **log** directory, which allows resuming functionality. 

In order to run all experiments, execute the corresponding *train.py* script:

```bash
# train.py resides under SRM/../code directory
python train.py
```

There is an option to enable the test mode to verify the setup, please modify the value of the *STAGE* variable in *exp_configurations.py* to 'test':

```python
# exp_configurations.py
STAGE = 'test'
``` 
Under this option, all the methods are trained with a few number of epochs and limited configurations. This is especially useful for imagenet experiments.  

To run all experiments in deploy mode, please set the *STAGE* variable to 'deploy':

```python
# exp_configurations.py
STAGE = 'deploy'
```

**Note1: all experiments were run with V100 32GB GPU so the batch size was set to fit the available memory. Please modify the batch size in exp_configurations.py to fit your GPU memory**
  
**Note2: except imagenet experiments, other experiments were run and implemented for the single GPU setup. For imagenet, we trained the methods on 2 V100. The number of GPUs and the GPU device indices can be set via the variable GPU_INDICES in exp_configurations.py**:

```python
# exp_configurations.py
GPU_INDICES = [0, 1] # using 2 GPU with device ids 0 and 1
```
 
## Pretrained ImageNet Models
We provide the pretrained ResNet18 model (top1 accuracy: **71.21**), which can be downloaded from [this](https://drive.google.com/file/d/1Dbj1AWzFeQGkFcdgSmClw3PkkSkhIXCu/view?usp=sharing). We also include a script (*eval.py*) that evaluates this pretrained model on the validation set of imagenet. In order to use this script, the pretrained model should be put under **SRM/data** directory:

```bash
# SRM/imagenet/code/
python eval.py
```


