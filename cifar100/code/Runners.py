import Utility
import exp_configurations as Conf
import Datasets
import torch
import os
import pickle
import DenseNet
import AllCNN
import numpy as np
import Layers



def get_class_count(dataset):
    if dataset == 'cifar100':
        return 100

    raise RuntimeError('unknown dataset')

def init_HintKD(args):

    result_file = '_'.join([str(v) for v in args]) + '.pickle'
    result_file = os.path.join(Conf.OUTPUT_DIR, result_file)
    if os.path.exists(result_file):
        fid = open(result_file, 'rb')
        result = pickle.load(fid)
        fid.close()
        return result


    prefix = args[0]
    dataset = args[1]
    scale = args[2]
    teacher_wd = args[3]
    student_wd = args[4]
    exp = args[5]

    """ check if teacher exists """

    t_args = ['teacher', dataset, teacher_wd]
    t_prefix = '_'.join([str(v) for v in t_args]) 
    t_files = [f for f in os.listdir(Conf.OUTPUT_DIR) if f.startswith(t_prefix)\
            and f.endswith('pickle')]
    
    # check if teacher model exists and use the best one according to val acc
    assert len(t_files) > 0
    t_file = None
    t_val_acc = 0
    for f in t_files:
        t_file_ = os.path.join(Conf.OUTPUT_DIR, f)
        fid = open(t_file_, 'rb')
        t_data = pickle.load(fid)
        fid.close()
        if t_data['val_acc'] > t_val_acc:
            t_val_acc = t_data['val_acc']
            t_file = t_file_

    fid = open(t_file, 'rb')
    t_data = pickle.load(fid)
    fid.close()


    data = Datasets.get_cifar100(batch_size=Conf.CODEWORD_BS)

    train_loader, val_loader, test_loader = data

    nb_class = get_class_count(dataset)

    T_model = DenseNet.HintDenseNet121(nb_class)
    
    T_model.load_state_dict(t_data['model_weights'])
    
    hint_dims = [128, 256, 1024] 
    S_model = AllCNN.HintAllCNN(hint_dims,
                                scale,
                                nb_class)


    logfile = '_'.join([str(v) for v in args]) + '.pickle'
    logfile = os.path.join(Conf.LOG_DIR, logfile)

    compute = 'gpu' if torch.cuda.is_available() else 'cpu'
    
    if Conf.STAGE == 'test':
        LR = Conf.INIT_TEST_LR
        Epochs = Conf.INIT_TEST_EPOCH
        verbose = True
    else:
        LR = Conf.INIT_LR
        Epochs = Conf.INIT_EPOCH
        verbose = False

    logdir = '_'.join([str(v) for v in args]) 
    logdir = os.path.join(Conf.LOG_DIR, logdir)
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    
    outputs = Utility.initialize_hint(logdir,
                                       S_model,
                                       T_model,
                                       compute,
                                       train_loader,
                                       LR,
                                       Epochs,
                                       student_wd,
                                       verbose)

       
    return outputs


def SRMKD(args):
   
    result_file = '_'.join([str(v) for v in args]) + '.pickle'
    result_file = os.path.join(Conf.OUTPUT_DIR, result_file)
    if os.path.exists(result_file):
        fid = open(result_file, 'rb')
        result = pickle.load(fid)
        fid.close()
        return result

    prefix = args[0]
    dataset = args[1]
    scale = args[2]
    codeword_multiplier = args[3]
    sparsity_multiplier = args[4]
    init_type = args[5]
    teacher_wd = args[6]
    student_wd = args[7]
    alpha = args[8]
    temperature = args[9]
    exp = args[10]

    """ check if teacher exists """

    t_args = ['teacher', dataset, teacher_wd]
    t_prefix = '_'.join([str(v) for v in t_args]) 
    t_files = [f for f in os.listdir(Conf.OUTPUT_DIR) if f.startswith(t_prefix)\
            and f.endswith('pickle')]
    
    # check if teacher model exists and use the best one according to val acc
    assert len(t_files) > 0
    t_file = None
    t_val_acc = 0
    for f in t_files:
        t_file_ = os.path.join(Conf.OUTPUT_DIR, f)
        fid = open(t_file_, 'rb')
        t_data = pickle.load(fid)
        fid.close()
        if t_data['val_acc'] > t_val_acc:
            t_val_acc = t_data['val_acc']
            t_file = t_file_

    fid = open(t_file, 'rb')
    t_data = pickle.load(fid)
    fid.close()
    

    """ check if initialized values exists """
    init_args = ['srm_init', dataset,  
            teacher_wd, student_wd, 
            codeword_multiplier, sparsity_multiplier,
            scale, init_type, exp]
    
    init_file = '_'.join([str(v) for v in init_args]) + '.pickle'
    init_file = os.path.join(Conf.OUTPUT_DIR, init_file)
    
    if os.path.exists(init_file):
        print('exist initialized values')
        fid = open(init_file, 'rb')
        init_data = pickle.load(fid)
        fid.close()
    else:
        print('need to train srm init')
        init_data = initialize_student(init_args)
        fid = open(init_file, 'wb')
        pickle.dump(init_data, fid)
        fid.close()
    
    
    data = Datasets.get_cifar100(batch_size=Conf.DISTIL_BS)

    train_loader, val_loader, test_loader = data

    nb_class = get_class_count(dataset)

    S_model = AllCNN.AllCNN(scale, nb_class)
    
    init_weights = init_data['model_weights']
    cur_weights = S_model.state_dict()
    
    for layer in cur_weights.keys():
        if layer in init_weights.keys():
            cur_weights[layer] = init_weights[layer]

    S_model.load_state_dict(cur_weights)


    T_model = DenseNet.DenseNet121(nb_class)
    T_model.load_state_dict(t_data['model_weights'])

    logfile = '_'.join([str(v) for v in args]) + '.pickle'
    logfile = os.path.join(Conf.LOG_DIR, logfile)

    compute = 'gpu' if torch.cuda.is_available() else 'cpu'
    
    if Conf.STAGE == 'test':
        LR = Conf.KD_TEST_LR
        Epochs = Conf.KD_TEST_EPOCH
        verbose = True
    else:
        LR = Conf.KD_LR
        Epochs = Conf.KD_EPOCH
        verbose = False
 

    print('train student using SRMKD')
    outputs = Utility.knowledge_distil(logfile,
                                       S_model,
                                       T_model,
                                       compute,
                                       train_loader,
                                       val_loader,
                                       test_loader,
                                       LR,
                                       Epochs,
                                       student_wd,
                                       alpha,
                                       temperature,
                                       verbose)
   
    return outputs

def SRMKDwT(args):
   
    result_file = '_'.join([str(v) for v in args]) + '.pickle'
    result_file = os.path.join(Conf.OUTPUT_DIR, result_file)
    if os.path.exists(result_file):
        fid = open(result_file, 'rb')
        result = pickle.load(fid)
        fid.close()
        return result

    prefix = args[0]
    dataset = args[1]
    scale = args[2]
    codeword_multiplier = args[3]
    sparsity_multiplier = args[4]
    target_type = args[5]
    teacher_wd = args[6]
    student_wd = args[7]
    exp = args[8]

    """ check if initialized values exists """
    init_args = ['srm_init', dataset,  
            teacher_wd, student_wd, 
            codeword_multiplier, sparsity_multiplier, 
            scale, target_type, exp]
    
    init_file = '_'.join([str(v) for v in init_args]) + '.pickle'
    init_file = os.path.join(Conf.OUTPUT_DIR, init_file)

    if os.path.exists(init_file):
        print('exist initialized values')
        fid = open(init_file, 'rb')
        init_data = pickle.load(fid)
        fid.close()
    else:
        print('need to train srm init')
        init_data = initialize_student(init_args)
        fid = open(init_file, 'wb')
        pickle.dump(init_data, fid)
        fid.close()
    
    
    data = Datasets.get_cifar100(batch_size=Conf.DISTIL_BS)

    train_loader, val_loader, test_loader = data

    nb_class = get_class_count(dataset)

    model = AllCNN.AllCNN(scale, nb_class)
    
    init_weights = init_data['model_weights']
    cur_weights = model.state_dict()
    
    for layer in cur_weights.keys():
        if layer in init_weights.keys():
            cur_weights[layer] = init_weights[layer]

    model.load_state_dict(cur_weights)
    
    logfile = '_'.join([str(v) for v in args]) + '.pickle'
    logfile = os.path.join(Conf.LOG_DIR, logfile)

    compute = 'gpu' if torch.cuda.is_available() else 'cpu'
    
    if Conf.STAGE == 'test':
        LR = Conf.KD_TEST_LR
        Epochs = Conf.KD_TEST_EPOCH
        verbose = True
    else:
        LR = Conf.KD_LR
        Epochs = Conf.KD_EPOCH
        verbose = False
 
    # train the whole network
    whole_outputs = Utility.train_classifier(logfile,
                                           model,
                                           compute,
                                           train_loader,
                                           val_loader,
                                           test_loader,
                                           LR,
                                           Epochs,
                                           student_wd,
                                           verbose)
   
    # train only output layer
    model.cpu()
    model.load_state_dict(cur_weights)

    if Conf.STAGE == 'test':
        LR = Conf.LINEAR_TEST_LR
        Epochs = Conf.LINEAR_TEST_EPOCH
        verbose = True
    else:
        LR = Conf.LINEAR_LR
        Epochs = Conf.LINEAR_EPOCH
        verbose = False

    linear_outputs = Utility.train_linear_classifier(logfile,
                                                     model,
                                                     compute,
                                                     train_loader,
                                                     val_loader,
                                                     test_loader,
                                                     LR,
                                                     Epochs,
                                                     student_wd,
                                                     verbose)


    outputs = {}
    
    outputs['whole_outputs'] = whole_outputs
    outputs['linear_outputs'] = linear_outputs

    outputs['train_acc_whole'] = whole_outputs['train_acc']
    outputs['val_acc_whole'] = whole_outputs['val_acc']
    outputs['test_acc_whole'] = whole_outputs['test_acc']
    
    outputs['train_acc_linear'] = linear_outputs['train_acc']
    outputs['val_acc_linear'] = linear_outputs['val_acc']
    outputs['test_acc_linear'] = linear_outputs['test_acc']
    
    return outputs


def HintKDwT(args):
   
    result_file = '_'.join([str(v) for v in args]) + '.pickle'
    result_file = os.path.join(Conf.OUTPUT_DIR, result_file)
    if os.path.exists(result_file):
        fid = open(result_file, 'rb')
        result = pickle.load(fid)
        fid.close()
        return result


    prefix = args[0]
    dataset = args[1]
    scale = args[2]
    teacher_wd = args[3]
    student_wd = args[4]
    exp = args[5]

    
    """ check if hints exists """
    hint_args = ['hint_init', dataset, scale, teacher_wd, student_wd, exp]
    hint_file = '_'.join([str(v) for v in hint_args]) + '.pickle'
    hint_file = os.path.join(Conf.OUTPUT_DIR, hint_file)

    
    if os.path.exists(hint_file):
        print('exist hints')
        fid = open(hint_file, 'rb')
        hint_data = pickle.load(fid)
        fid.close()
    else:
        print('need to train hints')
        hint_data = init_HintKD(hint_args)
        fid = open(hint_file, 'wb')
        pickle.dump(hint_data, fid)
        fid.close()
    
    
    data = Datasets.get_cifar100(batch_size=Conf.DISTIL_BS)

    train_loader, val_loader, test_loader = data

    nb_class = get_class_count(dataset)

    model = AllCNN.AllCNN(scale, nb_class)

    init_weights = hint_data['model_weights']
    cur_weights = model.state_dict()

    for key in cur_weights.keys():
        if key in init_weights.keys():
            cur_weights[key] = init_weights[key]


    model.load_state_dict(cur_weights)

    logfile = '_'.join([str(v) for v in args]) + '.pickle'
    logfile = os.path.join(Conf.LOG_DIR, logfile)

    compute = 'gpu' if torch.cuda.is_available() else 'cpu'
    
    if Conf.STAGE == 'test':
        LR = Conf.KD_TEST_LR
        Epochs = Conf.KD_TEST_EPOCH
        verbose = True
    else:
        LR = Conf.KD_LR
        Epochs = Conf.KD_EPOCH
        verbose = False
 
    whole_outputs = Utility.train_classifier(logfile,
                                           model,
                                           compute,
                                           train_loader,
                                           val_loader,
                                           test_loader,
                                           LR,
                                           Epochs,
                                           student_wd,
                                           verbose)
    
    
    # train only output layer
    model.cpu()
    model.load_state_dict(cur_weights)

    if Conf.STAGE == 'test':
        LR = Conf.LINEAR_TEST_LR
        Epochs = Conf.LINEAR_TEST_EPOCH
        verbose = True
    else:
        LR = Conf.LINEAR_LR
        Epochs = Conf.LINEAR_EPOCH
        verbose = False

    linear_outputs = Utility.train_linear_classifier(logfile,
                                                     model,
                                                     compute,
                                                     train_loader,
                                                     val_loader,
                                                     test_loader,
                                                     LR,
                                                     Epochs,
                                                     student_wd,
                                                     verbose)


    outputs = {}
    
    outputs['whole_outputs'] = whole_outputs
    outputs['linear_outputs'] = linear_outputs

    outputs['train_acc_whole'] = whole_outputs['train_acc']
    outputs['val_acc_whole'] = whole_outputs['val_acc']
    outputs['test_acc_whole'] = whole_outputs['test_acc']
    
    outputs['train_acc_linear'] = linear_outputs['train_acc']
    outputs['val_acc_linear'] = linear_outputs['val_acc']
    outputs['test_acc_linear'] = linear_outputs['test_acc']
    
    return outputs

    
    
    
    return outputs

def HintKD(args):
   
    result_file = '_'.join([str(v) for v in args]) + '.pickle'
    result_file = os.path.join(Conf.OUTPUT_DIR, result_file)
    if os.path.exists(result_file):
        fid = open(result_file, 'rb')
        result = pickle.load(fid)
        fid.close()
        return result


    prefix = args[0]
    dataset = args[1]
    scale = args[2]
    teacher_wd = args[3]
    student_wd = args[4]
    alpha = args[5]
    temperature = args[6]
    exp = args[7]

    """ check if teacher exists """
    t_args = ['teacher', dataset, teacher_wd]
    t_prefix = '_'.join([str(v) for v in t_args]) 
    t_files = [f for f in os.listdir(Conf.OUTPUT_DIR) if f.startswith(t_prefix)\
            and f.endswith('pickle')]

    # check if teacher model exists and use the best one according to val acc
    assert len(t_files) > 0
    t_file = None
    t_val_acc = 0
    for f in t_files:
        t_file_ = os.path.join(Conf.OUTPUT_DIR, f)
        fid = open(t_file_, 'rb')
        t_data = pickle.load(fid)
        fid.close()
        if t_data['val_acc'] > t_val_acc:
            t_val_acc = t_data['val_acc']
            t_file = t_file_

    fid = open(t_file, 'rb')
    t_data = pickle.load(fid)
    fid.close()


    """ check if hints exists """
    hint_args = ['hint_init', dataset, scale, teacher_wd, student_wd, exp]
    hint_file = '_'.join([str(v) for v in hint_args]) + '.pickle'
    hint_file = os.path.join(Conf.OUTPUT_DIR, hint_file)

    if os.path.exists(hint_file):
        print('exist hints')
        fid = open(hint_file, 'rb')
        hint_data = pickle.load(fid)
        fid.close()
    else:
        print('need to train hints')
        hint_data = init_HintKD(hint_args)
        fid = open(hint_file, 'wb')
        pickle.dump(hint_data, fid)
        fid.close()
    
    data = Datasets.get_cifar100(batch_size=Conf.DISTIL_BS)

    train_loader, val_loader, test_loader = data

    nb_class = get_class_count(dataset)

    S_model = AllCNN.AllCNN(scale, nb_class)

    init_weights = hint_data['model_weights']
    cur_weights = S_model.state_dict()

    for key in cur_weights.keys():
        if key in init_weights.keys():
            cur_weights[key] = init_weights[key]


    S_model.load_state_dict(cur_weights)


    T_model = DenseNet.DenseNet121(nb_class)
    T_model.load_state_dict(t_data['model_weights'])

    logfile = '_'.join([str(v) for v in args]) + '.pickle'
    logfile = os.path.join(Conf.LOG_DIR, logfile)

    compute = 'gpu' if torch.cuda.is_available() else 'cpu'
    
    if Conf.STAGE == 'test':
        LR = Conf.KD_TEST_LR
        Epochs = Conf.KD_TEST_EPOCH
        verbose = True
    else:
        LR = Conf.KD_LR
        Epochs = Conf.KD_EPOCH
        verbose = False
 

    outputs = Utility.knowledge_distil(logfile,
                                       S_model,
                                       T_model,
                                       compute,
                                       train_loader,
                                       val_loader,
                                       test_loader,
                                       LR,
                                       Epochs,
                                       student_wd,
                                       alpha,
                                       temperature,
                                       verbose)
    
    
    return outputs



def KD(args):
   
    result_file = '_'.join([str(v) for v in args]) + '.pickle'
    result_file = os.path.join(Conf.OUTPUT_DIR, result_file)
    if os.path.exists(result_file):
        fid = open(result_file, 'rb')
        result = pickle.load(fid)
        fid.close()
        return result


    prefix = args[0]
    dataset = args[1]
    scale = args[2]
    teacher_wd = args[3]
    student_wd = args[4]
    alpha = args[5]
    temperature = args[6]
    exp = args[7]

    """ check if teacher exists """

    t_args = ['teacher', dataset, teacher_wd]
    t_prefix = '_'.join([str(v) for v in t_args]) 
    t_files = [f for f in os.listdir(Conf.OUTPUT_DIR) if f.startswith(t_prefix)\
            and f.endswith('pickle')]

    # check if teacher model exists and use the best one according to val acc
    assert len(t_files) > 0
    t_file = None
    t_val_acc = 0
    for f in t_files:
        t_file_ = os.path.join(Conf.OUTPUT_DIR, f)
        fid = open(t_file_, 'rb')
        t_data = pickle.load(fid)
        fid.close()
        if t_data['val_acc'] > t_val_acc:
            t_val_acc = t_data['val_acc']
            t_file = t_file_

    fid = open(t_file, 'rb')
    t_data = pickle.load(fid)
    fid.close()
 

    data = Datasets.get_cifar100(batch_size=Conf.DISTIL_BS)

    train_loader, val_loader, test_loader = data

    nb_class = get_class_count(dataset)

    S_model = AllCNN.AllCNN(scale, nb_class)
    T_model = DenseNet.DenseNet121(nb_class)

    T_model.load_state_dict(t_data['model_weights'])

    logfile = '_'.join([str(v) for v in args]) + '.pickle'
    logfile = os.path.join(Conf.LOG_DIR, logfile)

    compute = 'gpu' if torch.cuda.is_available() else 'cpu'
    
    if Conf.STAGE == 'test':
        LR = Conf.KD_TEST_LR
        Epochs = Conf.KD_TEST_EPOCH
        verbose = True
    else:
        LR = Conf.KD_LR
        Epochs = Conf.KD_EPOCH
        verbose = False
 

    outputs = Utility.knowledge_distil(logfile,
                                       S_model,
                                       T_model,
                                       compute,
                                       train_loader,
                                       val_loader,
                                       test_loader,
                                       LR,
                                       Epochs,
                                       student_wd,
                                       alpha,
                                       temperature,
                                       verbose)
    

    return outputs



def train_student(args):
    result_file = '_'.join([str(v) for v in args]) + '.pickle'
    result_file = os.path.join(Conf.OUTPUT_DIR, result_file)
    if os.path.exists(result_file):
        fid = open(result_file, 'rb')
        result = pickle.load(fid)
        fid.close()
        return result


    prefix = args[0]
    dataset = args[1]
    scale = args[2]
    weight_decay = args[3]

    data = Datasets.get_cifar100(batch_size=Conf.ALLCNN_BS)

    train_loader, val_loader, test_loader = data

    nb_class = get_class_count(dataset)

    model = AllCNN.AllCNN(scale, nb_class)

    logfile = '_'.join([str(v) for v in args]) + '.pickle'
    logfile = os.path.join(Conf.LOG_DIR, logfile)

    compute = 'gpu' if torch.cuda.is_available() else 'cpu'
    
    if Conf.STAGE == 'test':
        LR = Conf.S_TEST_LR
        Epochs = Conf.S_TEST_EPOCH
        verbose = True
    else:
        LR = Conf.S_LR
        Epochs = Conf.S_EPOCH
        verbose = False
 

    outputs = Utility.train_classifier(logfile,
                                       model,
                                       compute,
                                       train_loader,
                                       val_loader,
                                       test_loader,
                                       LR,
                                       Epochs,
                                       weight_decay,
                                       verbose)


    return outputs



def train_teacher(args):
    

    result_file = '_'.join([str(v) for v in args]) + '.pickle'
    result_file = os.path.join(Conf.OUTPUT_DIR, result_file)
    if os.path.exists(result_file):
        fid = open(result_file, 'rb')
        result = pickle.load(fid)
        fid.close()
        return result


    prefix = args[0]
    dataset = args[1]
    weight_decay = args[2]


    data = Datasets.get_cifar100(batch_size=Conf.DENSENET_BS)

    train_loader, val_loader, test_loader = data

    nb_class = get_class_count(dataset)

    model = DenseNet.DenseNet121(nb_class)


    logfile = '_'.join([str(v) for v in args]) + '.pickle'
    logfile = os.path.join(Conf.LOG_DIR, logfile)

    compute = 'gpu' if torch.cuda.is_available() else 'cpu'
    
    if Conf.STAGE == 'test':
        LR = Conf.T_TEST_LR
        Epochs = Conf.T_TEST_EPOCH
        verbose = True
    else:
        LR = Conf.T_LR
        Epochs = Conf.T_EPOCH
        verbose = False
 

    outputs = Utility.train_classifier(logfile,
                                           model,
                                           compute,
                                           train_loader,
                                           val_loader,
                                           test_loader,
                                           LR,
                                           Epochs,
                                           weight_decay,
                                           verbose)
                                           

    
    return outputs

def train_codeword(args):
   
    
    result_file = '_'.join([str(v) for v in args]) + '.pickle'
    result_file = os.path.join(Conf.OUTPUT_DIR, result_file)
    if os.path.exists(result_file):
        fid = open(result_file, 'rb')
        result = pickle.load(fid)
        fid.close()
        return result


    prefix = args[0]
    dataset = args[1]
    teacher_wd = args[2]
    codeword_multiplier = args[3]
    sparsity_multiplier = args[4]
    exp = args[5]

    """ check if teacher exists """

    t_args = ['teacher', dataset, teacher_wd]
    t_prefix = '_'.join([str(v) for v in t_args]) 
    t_files = [f for f in os.listdir(Conf.OUTPUT_DIR) if f.startswith(t_prefix)\
            and f.endswith('pickle')]

    # check if teacher model exists and use the best one according to val acc
    assert len(t_files) > 0
    t_file = None
    t_val_acc = 0
    for f in t_files:
        t_file_ = os.path.join(Conf.OUTPUT_DIR, f)
        fid = open(t_file_, 'rb')
        t_data = pickle.load(fid)
        fid.close()
        if t_data['val_acc'] > t_val_acc:
            t_val_acc = t_data['val_acc']
            t_file = t_file_

    fid = open(t_file, 'rb')
    t_data = pickle.load(fid)
    fid.close()

    batch_size = Conf.CODEWORD_BS

    data = Datasets.get_cifar100(batch_size=batch_size)

    train_loader, val_loader, test_loader = data

    nb_class = get_class_count(dataset)

    compute = 'gpu' if torch.cuda.is_available() else 'cpu'

    model = DenseNet.SparseRepDenseNet121(codeword_multiplier, 
                                         sparsity_multiplier,
                                         centers=None,
                                         intercept=None,
                                         nb_class=nb_class)

    """ initialize with pretrained weights """ 
    cur_weights = model.state_dict()
    pretrained_weights = t_data['model_weights']

    for layer in cur_weights.keys():
        if layer in pretrained_weights.keys():
            cur_weights[layer] = pretrained_weights[layer]

    model.load_state_dict(cur_weights)


    logfile = '_'.join([str(v) for v in args]) + '.pickle'
    logfile = os.path.join(Conf.LOG_DIR, logfile)

        
    if Conf.STAGE == 'test':
        LR = Conf.CLUSTER_TEST_LR
        Epochs = Conf.CLUSTER_TEST_EPOCH
        verbose = True
    else:
        LR = Conf.CLUSTER_LR
        Epochs = Conf.CLUSTER_EPOCH
        verbose = False
 
    logdir = '_'.join([str(v) for v in args]) 
    logdir = os.path.join(Conf.LOG_DIR, logdir)
    if not os.path.exists(logdir):
        os.mkdir(logdir)

    outputs = Utility.train_codeword(logdir,
                                       model,
                                       compute,
                                       train_loader,
                                       LR,
                                       Epochs,
                                       0.0,
                                       verbose)


    return outputs


def initialize_student(args):

    result_file = '_'.join([str(v) for v in args]) + '.pickle'
    result_file = os.path.join(Conf.OUTPUT_DIR, result_file)
    if os.path.exists(result_file):
        fid = open(result_file, 'rb')
        result = pickle.load(fid)
        fid.close()
        return result


    prefix = args[0]
    dataset = args[1]
    teacher_wd = args[2]
    student_wd = args[3]
    codeword_multiplier = args[4]
    sparsity_multiplier = args[5]
    scale = args[6]
    target_type = args[7]
    exp = args[8]

    """ check if codeword exists """

    codeword_args = ['codeword', dataset, 
                     teacher_wd, codeword_multiplier, sparsity_multiplier,
                     exp]
    codeword_file = '_'.join([str(v) for v in codeword_args]) + '.pickle'
    codeword_file = os.path.join(Conf.OUTPUT_DIR, codeword_file)
    
    if os.path.exists(codeword_file):
        fid = open(codeword_file, 'rb')
        codeword_data = pickle.load(fid)
        fid.close()
    else:
        codeword_data = train_codeword(codeword_args)
        fid = open(codeword_file, 'wb')
        pickle.dump(codeword_data, fid)
        fid.close()

    """ load data """
    data = Datasets.get_cifar100(batch_size=Conf.HISTOGRAM_BS)

    train_loader, val_loader, test_loader = data

    nb_class = get_class_count(dataset)

    compute = 'gpu' if torch.cuda.is_available() else 'cpu'

    """ build the model """
    assert target_type in ['global-local', 'local', 'global']

    if target_type == 'global-local':
        S_model = AllCNN.GlobalLocalPred(codeword_multiplier, 
                                  centers=None,
                                  intercept=None,
                                  scale=scale,
                                  nb_class=nb_class)
        
        T_model = DenseNet.GlobalLocalLabel(codeword_multiplier,
                                       sparsity_multiplier,
                                       centers=None,
                                       intercept=None,
                                       nb_class=nb_class) 
    
    elif target_type == 'local':
        S_model = AllCNN.LocalPred(codeword_multiplier, 
                                  centers=None,
                                  intercept=None,
                                  scale=scale,
                                  nb_class=nb_class)
        
        T_model = DenseNet.LocalLabel(codeword_multiplier,
                                       sparsity_multiplier,
                                       centers=None,
                                       intercept=None,
                                       nb_class=nb_class)

    else:
        S_model = AllCNN.GlobalPred(codeword_multiplier, 
                                  centers=None,
                                  intercept=None,
                                  scale=scale,
                                  nb_class=nb_class)
        
        T_model = DenseNet.GlobalLabel(codeword_multiplier,
                                       sparsity_multiplier,
                                       centers=None,
                                       intercept=None,
                                       nb_class=nb_class)


    T_model.load_state_dict(codeword_data['model_weights'])

        
    if Conf.STAGE == 'test':
        LR = Conf.INIT_TEST_LR
        Epochs = Conf.INIT_TEST_EPOCH
        verbose = True
    else:
        LR = Conf.INIT_LR
        Epochs = Conf.INIT_EPOCH
        verbose = False
 

    logdir = '_'.join([str(v) for v in args]) 
    logdir = os.path.join(Conf.LOG_DIR, logdir)
   
    if not os.path.exists(logdir):
        os.mkdir(logdir)

    outputs = Utility.srm_init(logdir,
                               target_type,
                               S_model,
                               T_model,
                               compute,
                               train_loader,
                               val_loader,
                               test_loader,
                               LR,
                               Epochs,
                               student_wd,
                               verbose)

    return outputs


