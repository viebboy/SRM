import Utility
import exp_configurations as Conf
import Datasets
import torch
import os
import pickle
import numpy as np
import Layers
import Models


def get_class_count(dataset):
    if dataset == 'cub':
        return 200 
    if dataset == 'stanford_cars':
        return 196 
    if dataset == 'flower102':
        return 102
    if dataset == 'mit_indoor':
        return 67
    if dataset == 'pubfig83':
        return 83
    
    if dataset == 'cub-5':
        return 200 
    if dataset == 'stanford_cars-5':
        return 196 
    if dataset == 'flower102-5':
        return 102
    if dataset == 'mit_indoor-5':
        return 67
    if dataset == 'pubfig83-5':
        return 83

    if dataset == 'cub-10':
        return 200 
    if dataset == 'stanford_cars-10':
        return 196 
    if dataset == 'mit_indoor-10':
        return 67
    if dataset == 'pubfig83-10':
        return 83

    raise RuntimeError('unknown dataset')

def convert(name):
    if name.endswith('-5'):
        return name[:-2]
    if name.endswith('-10'):
        return name[:-3]

    return name

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
    teacher_model = args[2]
    student_model = args[3]
    teacher_wd = args[4]
    student_wd = args[5]
    exp = args[6]

    """ check if teacher exists """
    assert teacher_model in ['resnext50']

    t_args = ['teacher', 
              teacher_model, 
              convert(dataset), 
              teacher_wd]
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

    if student_model == 'allcnn':
        batch_size = Conf.HIST_ALLCNN_BS
    elif student_model == 'resnet18':
        batch_size = Conf.HIST_RESNET18_BS
    else:
        raise RuntimeError()

    batch_size *= len(Conf.GPU_INDICES)

    data = Datasets.get_data(dataset,
                             batch_size=batch_size)

    train_loader, val_loader, test_loader = data

    nb_class = get_class_count(dataset)

    if teacher_model == 'resnext50':
        T_model = Models.hint_label_resnext50(pretrained=True,
                                        nb_class=nb_class)
    else:
        raise RuntimeError()
    
    T_model.load_state_dict(t_data['model_weights'])

    if student_model == 'allcnn':
        S_model = Models.hint_pred_allcnn(T_model.hint_dims,
                                     Conf.ALLCNN_SCALE,
                                     nb_class)
    elif student_model == 'resnet18':
        S_model = Models.hint_pred_resnet18(T_model.hint_dims,
                                     nb_class)
    else:
        raise RuntimeError()

    if len(Conf.GPU_INDICES) > 1:
        S_model = Layers.CustomParallel(S_model, device_ids=Conf.GPU_INDICES)
        T_model = Layers.CustomParallel(T_model, device_ids=Conf.GPU_INDICES)
        multigpu = True
    else:
        multigpu = False

    device = torch.device('cuda')

    S_model.to(device)
    T_model.to(device)

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
                                       device,
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
    teacher_model = args[2]
    student_model = args[3]
    codeword_multiplier = args[4]
    sparsity_multiplier = args[5]
    target_type = args[6]
    teacher_wd = args[7]
    student_wd = args[8]
    alpha = args[9]
    temperature = args[10]
    exp = args[11]

    """ check if teacher exists """
    assert teacher_model in ['resnext50']

    t_args = ['teacher', 
              teacher_model, 
              convert(dataset), 
              teacher_wd]
    t_prefix = '_'.join([str(v) for v in t_args]) 
    
    t_files = [f for f in os.listdir(Conf.OUTPUT_DIR) if f.startswith(t_prefix)\
            and f.endswith('pickle')]

    # check if teacher model exists and use the best one according to val acc
    assert len(t_files) > 0, 'no teacher file exists'
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
            teacher_model, student_model, 
            teacher_wd, student_wd, 
            codeword_multiplier, sparsity_multiplier, 
            target_type, exp]
    
    init_file = '_'.join([str(v) for v in init_args]) + '.pickle'
    init_file = os.path.join(Conf.OUTPUT_DIR, init_file)

    
    if os.path.exists(init_file):
        print('exist initialized values')
        fid = open(init_file, 'rb')
        init_data = pickle.load(fid)
        fid.close()
    else:
        init_data = initialize_student(init_args)
        fid = open(init_file, 'wb')
        pickle.dump(init_data, fid)
        fid.close()
    
    if student_model == 'allcnn':
        batch_size = Conf.DISTIL_ALLCNN_BS
    elif student_model == 'resnet18':
        batch_size = Conf.DISTIL_RESNET18_BS
    else:
        raise RuntimeError('unknown student')
    
    batch_size *= len(Conf.GPU_INDICES)

    data = Datasets.get_data(dataset,
                             batch_size=batch_size)

    train_loader, val_loader, test_loader = data

    nb_class = get_class_count(dataset)

    if student_model == 'allcnn':
        S_model = Models.allcnn(scale=Conf.ALLCNN_SCALE, nb_class=nb_class)
    elif student_model == 'resnet18':
        S_model = Models.resnet18(pretrained=False, nb_class=nb_class)
    else:
        raise RuntimeError()

    
    init_weights = init_data['model_weights']
    cur_weights = S_model.state_dict()
    
    for layer in cur_weights.keys():
        if layer in init_weights.keys():
            cur_weights[layer] = init_weights[layer]

    print('loading student model weights')
    S_model.load_state_dict(cur_weights)

    if teacher_model == 'resnext50':
        T_model = Models.resnext50(pretrained=True,
                                   nb_class=nb_class)
    else:
        raise RuntimeError()

    print('loading teacher model weights')
    T_model.load_state_dict(t_data['model_weights'])

    if len(Conf.GPU_INDICES) > 1:
        S_model = Layers.CustomParallel(S_model, device_ids=Conf.GPU_INDICES)
        T_model = Layers.CustomParallel(T_model, device_ids=Conf.GPU_INDICES)

    device = torch.device('cuda')
    S_model.to(device)
    T_model.to(device)
    
    logfile = '_'.join([str(v) for v in args]) + '.pickle'
    logfile = os.path.join(Conf.LOG_DIR, logfile)
     
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
                                       device,
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
    teacher_model = args[2]
    student_model = args[3]
    teacher_wd = args[4]
    student_wd = args[5]
    alpha = args[6]
    temperature = args[7]
    exp = args[8]

    """ check if teacher exists """
    assert teacher_model in ['resnext50']

    t_args = ['teacher', 
              teacher_model, 
              convert(dataset), 
              teacher_wd]
    t_prefix = '_'.join([str(v) for v in t_args])

    t_files = [f for f in os.listdir(Conf.OUTPUT_DIR) if f.startswith(t_prefix)\
            and f.endswith('pickle')]

    # check if teacher model exists and use the best one according to val acc
    assert len(t_files) > 0, 'no teacher file exists'
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
    hint_args = ['hint_init', dataset, teacher_model, 
                 student_model, teacher_wd, student_wd, exp]
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
    
    if student_model == 'allcnn':
        batch_size = Conf.DISTIL_ALLCNN_BS
    elif student_model == 'resnet18':
        batch_size = Conf.DISTIL_RESNET18_BS
    else:
        raise RuntimeError('unknown student')
 
    batch_size *= len(Conf.GPU_INDICES)

    data = Datasets.get_data(dataset,
                             batch_size=batch_size)

    train_loader, val_loader, test_loader = data

    nb_class = get_class_count(dataset)

    if student_model == 'allcnn':
        S_model = Models.allcnn(scale=Conf.ALLCNN_SCALE, nb_class=nb_class)
    elif student_model == 'resnet18':
        S_model = Models.resnet18(pretrained=False, nb_class=nb_class)
    else:
        raise RuntimeError()

    init_weights = hint_data['model_weights']
    cur_weights = S_model.state_dict()

    for key in cur_weights.keys():
        if key in init_weights.keys():
            cur_weights[key] = init_weights[key]

    S_model.load_state_dict(cur_weights)

    if teacher_model == 'resnext50':
        T_model = Models.resnext50(pretrained=True,
                                   nb_class=nb_class)
    else:
        raise RuntimeError()

    print('loading teacher model weights')
    T_model.load_state_dict(t_data['model_weights'])

    if len(Conf.GPU_INDICES) > 1:
        S_model = Layers.CustomParallel(S_model, device_ids=Conf.GPU_INDICES)
        T_model = Layers.CustomParallel(T_model, device_ids=Conf.GPU_INDICES)

    device = torch.device('cuda')
    S_model.to(device)
    T_model.to(device)
 
    logfile = '_'.join([str(v) for v in args]) + '.pickle'
    logfile = os.path.join(Conf.LOG_DIR, logfile)

    
    if Conf.STAGE == 'test':
        LR = Conf.KD_TEST_LR
        Epochs = Conf.KD_TEST_EPOCH
        verbose = True
    else:
        LR = Conf.KD_LR
        Epochs = Conf.KD_EPOCH
        verbose = False
 

    print('train student using Hint KD')
    outputs = Utility.knowledge_distil(logfile,
                                       S_model,
                                       T_model,
                                       device,
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
    teacher_model = args[2]
    student_model = args[3]
    teacher_wd = args[4]
    student_wd = args[5]
    alpha = args[6]
    temperature = args[7]
    exp = args[8]

    """ check if teacher exists """
    assert teacher_model in ['resnext50']

    t_args = ['teacher', 
              teacher_model, 
              convert(dataset), 
              teacher_wd]
    t_prefix = '_'.join([str(v) for v in t_args])

    t_files = [f for f in os.listdir(Conf.OUTPUT_DIR) if f.startswith(t_prefix)\
            and f.endswith('pickle')]

    # check if teacher model exists and use the best one according to val acc
    assert len(t_files) > 0, 'no teacher file exists'
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

    if student_model == 'allcnn':
        batch_size = Conf.DISTIL_ALLCNN_BS
    elif student_model == 'resnet18':
        batch_size = Conf.DISTIL_RESNET18_BS
    else:
        raise RuntimeError('unknown student')

    batch_size *= len(Conf.GPU_INDICES)


    data = Datasets.get_data(dataset,
                             batch_size=batch_size)

    train_loader, val_loader, test_loader = data

    nb_class = get_class_count(dataset)

    if student_model == 'allcnn':
        S_model = Models.allcnn(scale=Conf.ALLCNN_SCALE, nb_class=nb_class)
    elif student_model == 'resnet18':
        S_model = Models.resnet18(pretrained=False, nb_class=nb_class)
    else:
        raise RuntimeError()

    if teacher_model == 'resnext50':
        T_model = Models.resnext50(pretrained=True,
                                   nb_class=nb_class)
    else:
        raise RuntimeError()

    print('loading teacher model weights')
    T_model.load_state_dict(t_data['model_weights'])

    if len(Conf.GPU_INDICES) > 1:
        S_model = Layers.CustomParallel(S_model, device_ids=Conf.GPU_INDICES)
        T_model = Layers.CustomParallel(T_model, device_ids=Conf.GPU_INDICES)

    device = torch.device('cuda')
    S_model.to(device)
    T_model.to(device) 

    logfile = '_'.join([str(v) for v in args]) + '.pickle'
    logfile = os.path.join(Conf.LOG_DIR, logfile)

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
                                       device,
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
    student_model = args[2]
    weight_decay = args[3]
    exp = args[4]

    
    if student_model == 'allcnn':
        batch_size = Conf.ALLCNN_BS
    elif student_model == 'resnet18':
        batch_size = Conf.RESNET18_BS
    else:
        raise RuntimeError()
    
    batch_size *= len(Conf.GPU_INDICES)

    data = Datasets.get_data(dataset,
                             batch_size=batch_size)

    train_loader, val_loader, test_loader = data

    nb_class = get_class_count(dataset)

    if student_model == 'allcnn':
        model = Models.allcnn(Conf.ALLCNN_SCALE, nb_class)
    elif student_model == 'resnet18':
        model = Models.resnet18(pretrained=False, nb_class=nb_class)
    else:
        raise RuntimeError()

    if len(Conf.GPU_INDICES) > 1:
        model = Layers.CustomParallel(model, device_ids=Conf.GPU_INDICES)

    device = torch.device('cuda')
    model.to(device)

    logdir = '_'.join([str(v) for v in args]) 
    logdir = os.path.join(Conf.LOG_DIR, logdir)

    if not os.path.exists(logdir):
        os.mkdir(logdir)
    
    logfile = os.path.join(logdir, 'logfile.pickle')

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
                                       device,
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
    teacher_model = args[1]
    dataset = args[2]
    weight_decay = args[3]
    exp = args[4]

    assert teacher_model in ['resnext50']

    if teacher_model == 'resnext50':
        batch_size = Conf.RESNEXT50_BS
    else:
        raise RuntimeError('unknown teacher model: %s' % str(teacher_model))

    batch_size *= len(Conf.GPU_INDICES)

    data = Datasets.get_data(dataset,
                             batch_size=batch_size)

    train_loader, val_loader, test_loader = data

    nb_class = get_class_count(dataset)

    if teacher_model == 'resnext50':
        model = Models.resnext50(pretrained=True, nb_class=nb_class)
    else:
        raise RuntimeError()

    if len(Conf.GPU_INDICES) > 1:
        model = Layers.CustomParallel(model, device_ids=Conf.GPU_INDICES)
    
    device = torch.device('cuda')

    logdir = '_'.join([str(v) for v in args])
    logdir = os.path.join(Conf.LOG_DIR, logdir)
    if not os.path.exists(logdir):
        os.mkdir(logdir)

    logfile = os.path.join(logdir, 'logfile.pickle')

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
                                           device,
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
    teacher_model = args[2]
    teacher_wd = args[3]
    codeword_multiplier = args[4]
    sparsity_multiplier = args[5]
    exp = args[6]

    assert teacher_model in ['resnext50']
    
    t_args = ['teacher', 
              teacher_model, 
              convert(dataset), 
              teacher_wd]
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

    """ load data """
    if teacher_model == 'resnext50':
        batch_size = Conf.CW_RESNEXT50_BS
    else:
        raise RuntimeError()
    
    batch_size *= len(Conf.GPU_INDICES)

    multigpu = True if len(Conf.GPU_INDICES) > 1 else False

    data = Datasets.get_data(dataset,
                             batch_size=batch_size)

    train_loader, val_loader, test_loader = data

    nb_class = get_class_count(dataset)


    if teacher_model == 'resnext50':
        model = Models.sparse_rep_resnext50(codeword_multiplier,
                                          sparsity_multiplier,
                                          nb_class,
                                          pretrained=True)
    else:
        raise RuntimeError()

    cur_weights = model.state_dict()
    pretrained_weights = t_data['model_weights']
    count = 0 
    for layer in cur_weights.keys():
        if layer in pretrained_weights.keys():
            cur_weights[layer] = pretrained_weights[layer]
            count += 1
        else:
            print('skipping layer: %s' % str(layer))

    model.load_state_dict(cur_weights)
    print('load weights from %s layers' % str(count))

    if len(Conf.GPU_INDICES) > 1:
        model = Layers.CustomParallel(model, device_ids=Conf.GPU_INDICES)

    device = torch.device('cuda')
    model.to(device)

    
    logdir = '_'.join([str(v) for v in args]) 
    logdir = os.path.join(Conf.LOG_DIR, logdir)
    if not os.path.exists(logdir):
        os.mkdir(logdir)

        
    logfile = os.path.join(logdir, 'logfile.pickle')
    model_file = os.path.join(logdir, 'best.pickle')

    if Conf.STAGE == 'test':
        LR = Conf.CLUSTER_TEST_LR
        Epochs = Conf.CLUSTER_TEST_EPOCH
        verbose = True
    else:
        LR = Conf.CLUSTER_LR
        Epochs = Conf.CLUSTER_EPOCH
        verbose = False
 
    outputs = Utility.train_codeword(logfile,
                                      model_file,
                                       model,
                                       multigpu,
                                       device,
                                       train_loader,
                                       LR,
                                       Epochs,
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
    teacher_model = args[2]
    student_model = args[3]
    teacher_wd = args[4]
    student_wd = args[5]
    codeword_multiplier = args[6]
    sparsity_multiplier = args[7]
    target_type = args[8]
    exp = args[9]

    assert teacher_model in ['resnext50']

    """ check if codeword exists """

    codeword_args = ['codeword', dataset, teacher_model,  
                     teacher_wd, codeword_multiplier, sparsity_multiplier, 
                     exp]
    
    codeword_file = '_'.join([str(v) for v in codeword_args]) + '.pickle'
    codeword_file = os.path.join(Conf.OUTPUT_DIR, codeword_file)
    print(codeword_file)

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
    if student_model == 'allcnn':
        batch_size = Conf.HIST_ALLCNN_BS
    elif student_model == 'resnet18':
        batch_size = Conf.HIST_RESNET18_BS
    else:
        raise RuntimeError()

    batch_size *= len(Conf.GPU_INDICES)

    data = Datasets.get_data(dataset,
                             batch_size=batch_size)

    train_loader, val_loader, test_loader = data

    nb_class = get_class_count(dataset)

    """ build the model """
    assert target_type in ['global-local']

    if teacher_model == 'resnext50':
        T_model = Models.label_resnext50(codeword_multiplier,
                                         sparsity_multiplier,
                                         nb_class,
                                         pretrained=True)
    else:
        raise RuntimeError()

    if student_model == 'allcnn':
        S_model = Models.pred_allcnn(T_model.nb_codewords,
                                     Conf.ALLCNN_SCALE,
                                     nb_class)

    elif student_model == 'resnet18':
        S_model = Models.pred_resnet18(T_model.nb_codewords,
                                       nb_class)
    else:
        raise RuntimeError()
    
    T_model.load_state_dict(codeword_data['model_weights'])
    
    if len(Conf.GPU_INDICES) > 1:
        S_model = Layers.CustomParallel(S_model, device_ids=Conf.GPU_INDICES)
        T_model = Layers.CustomParallel(T_model, device_ids=Conf.GPU_INDICES)
        multigpu = True
    else:
        multigpu = False

    device = torch.device('cuda')
    S_model.to(device)
    T_model.to(device)

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
                               device,
                               train_loader,
                               val_loader,
                               test_loader,
                               LR,
                               Epochs,
                               student_wd,
                               verbose)

    

    return outputs



