"""
the general training framework
"""

from __future__ import print_function

import os
import sys, getopt
import argparse
import socket
import time

import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np


from models import model_dict
from models.util import Embed, ConvReg, LinearEmbed
from models.util import Connector, Translator, Paraphraser

from dataset.datasets import get_dataloaders, get_dataloaders_sample

from helper.util import adjust_learning_rate

from distiller_zoo import DistillKL, HintLoss, Attention, Similarity, Correlation, VIDLoss, RKDLoss
from distiller_zoo import PKT, ABLoss, FactorTransfer, KDSVD, FSP, NSTLoss
from crd.criterion import CRDLoss

from helper.loops import train_distill as train, validate
from helper.pretrain import init
import pickle
import exp_configurations as Conf


def parse_option(dataset, method, trial, student):
    if dataset == 'cifar100':
        teacher = 'densenet121'
        batch_size = 128 
        init_epoch = 120
        decay_lr = '30,131' 
    else:
        teacher = 'resnext50'
        if method == 'nst':
            batch_size = 16
        else:
            batch_size = 128 
        init_epoch = 120
        decay_lr = '40,160'

    epochs = 200

    
    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=batch_size, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=epochs, help='number of training epochs')
    parser.add_argument('--init_epochs', type=int, default=init_epoch, help='init training for two-stage methods')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default=decay_lr, help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--dataset', type=str, default=dataset, help='dataset')

    # model
    parser.add_argument('--model_s', type=str, default=student,
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'ResNet50',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2', 'allcnn', 'allcnnt',
                                 'resnet18'])
    parser.add_argument('--path_t', type=str, default=teacher, help='teacher model snapshot')

    # distillation
    parser.add_argument('--distill', type=str, default=method, choices=['kd', 'hint', 'attention', 'similarity',
                                                                      'correlation', 'vid', 'crd', 'kdsvd', 'fsp',
                                                                      'rkd', 'pkt', 'abound', 'factor', 'nst'])
    parser.add_argument('--trial', type=str, default=str(trial), help='trial id')

    parser.add_argument('-r', '--gamma', type=float, default=1, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float, default=None, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=None, help='weight balance for other losses')

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')

    # NCE distillation
    parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'])
    parser.add_argument('--nce_k', default=16384, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')

    # hint layer
    parser.add_argument('--hint_layer', default=2, type=int, choices=[0, 1, 2, 3, 4])
    parser.add_argument('--index', default=0, type=int)

    try:
        opt = parser.parse_args()
    except:
        opt = parser.parse_args([])

    # set different learning rate from these 4 models
    if opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    # set the path according to the environment
    if hostname.startswith('visiongpu'):
        opt.model_path = '/path/to/my/student_model'
        opt.tb_path = '/path/to/my/student_tensorboards'
    else:
        opt.model_path = './save/student_model'
        opt.tb_path = './save/student_tensorboards'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_t = get_teacher_name(opt.path_t)

    opt.model_name = 'S:{}_T:{}_{}_{}_r:{}_a:{}_b:{}_{}'.format(opt.model_s, opt.model_t, opt.dataset, opt.distill,
                                                                opt.gamma, opt.alpha, opt.beta, opt.trial)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def get_teacher_name(model_path):
    """parse teacher name"""
    if '.pth' in model_path:
        segments = model_path.split('/')[-2].split('_')
        if segments[0] != 'wrn':
            return segments[0]
        else:
            return segments[0] + '_' + segments[1] + '_' + segments[2]
    else:
        return model_path


def load_teacher(dataset, model_path, n_cls):
    if '.pth' in model_path:
        print('==> loading teacher model')
        model_t = get_teacher_name(model_path)
        model = model_dict[model_t](num_classes=n_cls)
        model.load_state_dict(torch.load(model_path)['model'])
        print('==> done')
    else:
        model = model_dict[model_path](nb_class=n_cls) 
        fid = open('save/models/{}_{}.pickle'.format(dataset, model_path), 'rb')
        weights = pickle.load(fid)['model_weights']
        fid.close()
        model.load_state_dict(weights)
    
    return model


def load_student(name, nb_class):
    try:
        model = model_dict[name](num_classes=nb_class)
    except:
        model = model_dict[name](nb_class=nb_class)

    return model

def main(argv):
    
   
    """ parse index of experiment"""
    try:
      opts, args = getopt.getopt(argv,"h", ['index=', ])

    except getopt.GetoptError:
        sys.exit(2)

    for opt, arg in opts:
        if opt == '--index':
            index = int(arg)



    fid = open('missing.pickle', 'rb')
    leftover = pickle.load(fid)['configurations']
    fid.close()

    index = len(leftover) - index - 1

    _dataset, _method, _trial, _student = leftover[index]
    outputs = execute([_dataset, _method, _trial, _student])

    filename = 'output/{}_{}_{}_{}.pickle'.format(_dataset, _method, _trial, _student)
    fid = open(filename, 'wb')
    pickle.dump(outputs, fid)
    fid.close()


def execute(args):
    _dataset, _method, _trial, _student = args
    filename = 'output/{}_{}_{}_{}.pickle'.format(_dataset, _method, _trial, _student)
    if os.path.exists(filename):
        fid = open(filename, 'rb')
        outputs = pickle.load(fid)
        fid.close()
        return outputs

    best_acc = 0
    opt = parse_option(_dataset, _method, _trial, _student)
    opt.dataset = _dataset
    opt.distill = _method
    opt.trial = _trial
    opt.gamma = Conf.method_conf[_method]['r']
    opt.alpha = Conf.method_conf[_method]['a']
    opt.beta = Conf.method_conf[_method]['b']

    # if test mode, reduce epochs
    if Conf.mode == 'test':
        opt.epochs = 3
        opt.init_epochs = 3
        opt.lr_decay_epochs = [1, 2]
    

    # tensorboard logger
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # dataloader
    if opt.distill in ['crd']:
        train_loader, val_loader, test_loader, n_data, n_cls = get_dataloaders_sample(_dataset, batch_size=opt.batch_size,
                                                                           num_workers=opt.num_workers,
                                                                           k=opt.nce_k,
                                                                           mode=opt.mode)
    else:
        train_loader, val_loader, test_loader, n_data, n_cls = get_dataloaders(_dataset, batch_size=opt.batch_size,
                                                                    num_workers=opt.num_workers,
                                                                    is_instance=True)

    # model
    model_t = load_teacher(opt.dataset, opt.path_t, n_cls)
    model_s = load_student(opt.model_s, n_cls) 

    if _dataset == 'cifar100':
        data = torch.randn(2, 3, 32, 32)
    else:
        data = torch.randn(2, 3, 224, 224)

    model_t.eval()
    model_s.eval()
    feat_t, _ = model_t(data, is_feat=True)
    feat_s, _ = model_s(data, is_feat=True)

    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(opt.kd_T)
    if opt.distill == 'kd':
        criterion_kd = DistillKL(opt.kd_T)
    elif opt.distill == 'hint':
        criterion_kd = HintLoss()
        regress_s = ConvReg(feat_s[opt.hint_layer].shape, feat_t[opt.hint_layer].shape)
        module_list.append(regress_s)
        trainable_list.append(regress_s)
    elif opt.distill == 'crd':
        opt.s_dim = feat_s[-1].shape[1]
        opt.t_dim = feat_t[-1].shape[1]
        opt.n_data = n_data
        criterion_kd = CRDLoss(opt)
        module_list.append(criterion_kd.embed_s)
        module_list.append(criterion_kd.embed_t)
        trainable_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_t)
    elif opt.distill == 'attention':
        criterion_kd = Attention()
    elif opt.distill == 'nst':
        criterion_kd = NSTLoss()
    elif opt.distill == 'similarity':
        criterion_kd = Similarity()
    elif opt.distill == 'rkd':
        criterion_kd = RKDLoss()
    elif opt.distill == 'pkt':
        criterion_kd = PKT()
    elif opt.distill == 'kdsvd':
        criterion_kd = KDSVD()
    elif opt.distill == 'correlation':
        criterion_kd = Correlation()
        embed_s = LinearEmbed(feat_s[-1].shape[1], opt.feat_dim)
        embed_t = LinearEmbed(feat_t[-1].shape[1], opt.feat_dim)
        module_list.append(embed_s)
        module_list.append(embed_t)
        trainable_list.append(embed_s)
        trainable_list.append(embed_t)
    elif opt.distill == 'vid':
        s_n = [f.shape[1] for f in feat_s[1:-1]]
        t_n = [f.shape[1] for f in feat_t[1:-1]]
        criterion_kd = nn.ModuleList(
            [VIDLoss(s, t, t) for s, t in zip(s_n, t_n)]
        )
        # add this as some parameters in VIDLoss need to be updated
        trainable_list.append(criterion_kd)
    elif opt.distill == 'abound':
        s_shapes = [f.shape for f in feat_s[1:-1]]
        t_shapes = [f.shape for f in feat_t[1:-1]]
        connector = Connector(s_shapes, t_shapes)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(connector)
        init_trainable_list.append(model_s.get_feat_modules())
        criterion_kd = ABLoss(len(feat_s[1:-1]))
        init(model_s, model_t, init_trainable_list, criterion_kd, train_loader, logger, opt)
        # classification
        module_list.append(connector)
    elif opt.distill == 'factor':
        s_shape = feat_s[-2].shape
        t_shape = feat_t[-2].shape
        paraphraser = Paraphraser(t_shape)
        translator = Translator(s_shape, t_shape)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(paraphraser)
        criterion_init = nn.MSELoss()
        init(model_s, model_t, init_trainable_list, criterion_init, train_loader, logger, opt)
        # classification
        criterion_kd = FactorTransfer()
        module_list.append(translator)
        module_list.append(paraphraser)
        trainable_list.append(translator)
    elif opt.distill == 'fsp':
        s_shapes = [s.shape for s in feat_s[:-1]]
        t_shapes = [t.shape for t in feat_t[:-1]]
        criterion_kd = FSP(s_shapes, t_shapes)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(model_s.get_feat_modules())
        init(model_s, model_t, init_trainable_list, criterion_kd, train_loader, logger, opt)
        # classification training
        pass
    else:
        raise NotImplementedError(opt.distill)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # classification loss
    criterion_list.append(criterion_div)    # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)     # other knowledge distillation loss

    # optimizer
    optimizer = optim.Adam(trainable_list.parameters(),
                          lr=opt.learning_rate,
                          weight_decay=opt.weight_decay)

    # append teacher after optimizer to avoid weight_decay
    module_list.append(model_t)

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        cudnn.benchmark = True

    # validate teacher accuracy
    teacher_acc, _, _ = validate(test_loader, model_t, criterion_cls, opt)
    print('teacher accuracy: ', teacher_acc)

    # routine
    train_acc_list = []
    val_acc_list = []
    test_acc_list = []


    for epoch in range(1, opt.epochs + 1):

        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        train_acc, train_loss = train(epoch, train_loader, module_list, criterion_list, optimizer, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_loss', train_loss, epoch)

        val_acc, val_acc_top5, val_loss = validate(val_loader, model_s, criterion_cls, opt)

        logger.log_value('val_acc', val_acc, epoch)
        logger.log_value('val_loss', val_loss, epoch)
        logger.log_value('val_acc_top5', val_acc_top5, epoch)

        test_acc, tect_acc_top5, test_loss = validate(test_loader, model_s, criterion_cls, opt)

        logger.log_value('test_acc', test_acc, epoch)
        logger.log_value('test_loss', test_loss, epoch)
        logger.log_value('test_acc_top5', tect_acc_top5, epoch)

        train_acc_list.append(train_acc.item())
        val_acc_list.append(val_acc.item())
        test_acc_list.append(test_acc.item())

        # save the best model
        if val_acc > best_acc:
            best_acc = val_acc
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'best_acc': best_acc,
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model_s))
            print('saving the best model!')
            torch.save(state, save_file)

        # regular saving
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'accuracy': test_acc,
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

    # This best accuracy is only for printing purpose.
    # The results reported in the paper/README is from the last epoch. 
    print('best accuracy:', best_acc)

    # save model
    state = {
        'opt': opt,
        'model': model_s.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model_s))
    torch.save(state, save_file)

    outputs = {'train_acc': train_acc_list,
               'val_acc': val_acc_list,
               'test_acc': test_acc_list,
               'final_test_acc': test_acc_list[np.argmax(val_acc_list)],
               'weights': model_s.cpu().state_dict()} 

    return outputs

if __name__ == "__main__":
    main(sys.argv[1:])

