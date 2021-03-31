import os
import mxnet as mx
import torch
from mxnet.gluon.data.vision import transforms
from mxnet.gluon.data.vision.datasets import ImageRecordDataset
from mxnet import gluon
from gluoncv.data import transforms as gcv_transforms
import numpy as np
import exp_configurations as Conf


class _Dataset:

    def __init__(self, data_iter, length, steps):
        self.data_iter = data_iter
        self.count = 0
        self.total_samples = length
        self.steps = steps

    def __len__(self,):
        return self.steps

    def __iter__(self):
        return self

    def __next__(self):
        if self.count >= self.total_samples:
            self.data_iter.reset()
            self.count = 0
            raise StopIteration
        
        batch = next(self.data_iter)
        batch_size = batch.data[0].shape[0]

        batch_size = min(batch_size, self.total_samples - self.count)
        x = batch.data[0][:batch_size]
        y = batch.label[0][:batch_size]
       
        self.count += batch_size

        return torch.tensor(x.asnumpy()).float(), \
                torch.tensor(y.asnumpy()).long()


def check_length(filename):
    fid = open(filename, 'r')
    length = len(fid.read().split('\n')[:-1])
    fid.close()

    return length


def imagenet2012_subset(batch_size=128):
    
    workers = Conf.WORKERS
    input_shape = (3, 224, 224)

    train_rec = os.path.join(Conf.DATA_DIR, 'imagenet_train.rec') 
    train_idx = os.path.join(Conf.DATA_DIR, 'imagenet_train.idx') 
    val_rec = os.path.join(Conf.DATA_DIR, 'imagenet_val.rec') 
    val_idx = os.path.join(Conf.DATA_DIR, 'imagenet_val.idx')

    assert os.path.exists(train_rec), 'missing {}'.format(train_rec)
    assert os.path.exists(train_idx), 'missing {}'.format(train_idx)
    assert os.path.exists(val_rec), 'missing {}'.format(val_rec)
    assert os.path.exists(val_idx), 'missing {}'.format(val_idx)
    
    jitter_param = 0.4
    lighting_param = 0.1
    mean_rgb = [123.68, 116.779, 103.939]
    std_rgb = [58.393, 57.12, 57.375]

    train_data = mx.io.ImageRecordIter(
        path_imgrec         = train_rec,
        path_imgidx         = train_idx,
        preprocess_threads  = workers,
        round_batch         = 0,
        shuffle             = True,
        batch_size          = batch_size,
        data_shape          = input_shape,
        mean_r              = mean_rgb[0],
        mean_g              = mean_rgb[1],
        mean_b              = mean_rgb[2],
        std_r               = std_rgb[0],
        std_g               = std_rgb[1],
        std_b               = std_rgb[2],
        rand_mirror         = True,
        random_resized_crop = True,
        max_aspect_ratio    = 4. / 3.,
        min_aspect_ratio    = 3. / 4.,
        max_random_area     = 1,
        min_random_area     = 0.08,
        brightness          = jitter_param,
        saturation          = jitter_param,
        contrast            = jitter_param,
        pca_noise           = lighting_param,
        ctx                 = 'cpu'
        )


    val_data = mx.io.ImageRecordIter(
        path_imgrec         = val_rec,
        path_imgidx         = val_idx,
        preprocess_threads  = workers,
        shuffle             = False,
        batch_size          = batch_size,
        round_batch         = 0,
        resize              = 256,
        data_shape          = input_shape,
        mean_r              = mean_rgb[0],
        mean_g              = mean_rgb[1],
        mean_b              = mean_rgb[2],
        std_r               = std_rgb[0],
        std_g               = std_rgb[1],
        std_b               = std_rgb[2],
        ctx                 = 'cpu'
    )
    
    n_train = int(check_length(train_idx)/100)
    n_val = int(check_length(val_idx)/10)
    train_steps = int(np.ceil(n_train / float(batch_size)))
    val_steps = int(np.ceil(n_val / float(batch_size)))

    train_data = _Dataset(train_data, n_train, train_steps)
    val_data = _Dataset(val_data, n_val, val_steps)

    return train_data, val_data


def imagenet2012(batch_size=128):

    workers = Conf.WORKERS
    input_shape = (3, 224, 224)

    train_rec = os.path.join(Conf.DATA_DIR, 'imagenet_train.rec') 
    train_idx = os.path.join(Conf.DATA_DIR, 'imagenet_train.idx') 
    val_rec = os.path.join(Conf.DATA_DIR, 'imagenet_val.rec') 
    val_idx = os.path.join(Conf.DATA_DIR, 'imagenet_val.idx')

    assert os.path.exists(train_rec), 'missing {}'.format(train_rec)
    assert os.path.exists(train_idx), 'missing {}'.format(train_idx)
    assert os.path.exists(val_rec), 'missing {}'.format(val_rec)
    assert os.path.exists(val_idx), 'missing {}'.format(val_idx)
    
    jitter_param = 0.4
    lighting_param = 0.1
    mean_rgb = [123.68, 116.779, 103.939]
    std_rgb = [58.393, 57.12, 57.375]

    train_data = mx.io.ImageRecordIter(
        path_imgrec         = train_rec,
        path_imgidx         = train_idx,
        preprocess_threads  = workers,
        round_batch         = 0,
        shuffle             = True,
        batch_size          = batch_size,
        data_shape          = input_shape,
        mean_r              = mean_rgb[0],
        mean_g              = mean_rgb[1],
        mean_b              = mean_rgb[2],
        std_r               = std_rgb[0],
        std_g               = std_rgb[1],
        std_b               = std_rgb[2],
        rand_mirror         = True,
        random_resized_crop = True,
        max_aspect_ratio    = 4. / 3.,
        min_aspect_ratio    = 3. / 4.,
        max_random_area     = 1,
        min_random_area     = 0.08,
        brightness          = jitter_param,
        saturation          = jitter_param,
        contrast            = jitter_param,
        pca_noise           = lighting_param,
        ctx                 = 'cpu'
        )


    val_data = mx.io.ImageRecordIter(
        path_imgrec         = val_rec,
        path_imgidx         = val_idx,
        preprocess_threads  = workers,
        shuffle             = False,
        batch_size          = batch_size,
        round_batch         = 0,
        resize              = 256,
        data_shape          = input_shape,
        mean_r              = mean_rgb[0],
        mean_g              = mean_rgb[1],
        mean_b              = mean_rgb[2],
        std_r               = std_rgb[0],
        std_g               = std_rgb[1],
        std_b               = std_rgb[2],
        ctx                 = 'cpu'
    )


    n_train = check_length(train_idx)
    n_val = check_length(val_idx)
    train_steps = int(np.ceil(n_train / float(batch_size)))
    val_steps = int(np.ceil(n_val / float(batch_size)))

    train_data = _Dataset(train_data, n_train, train_steps)
    val_data = _Dataset(val_data, n_val, val_steps)

    return train_data, val_data


def get_data(dataset, 
             batch_size, 
             test_mode=False):
    
    if dataset == 'imagenet':
        if test_mode:
            return imagenet2012_subset(batch_size)
        else:
            return imagenet2012(batch_size)
    else:
        raise RuntimeError('unknown dataset')

