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
    if dataset == 'imagenet':
        return 1000

    raise RuntimeError('unknown dataset')


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
    codeword_multiplier = args[2]
    sparsity_multiplier = args[3]
    target_type = args[4]
    teacher_model = args[5]
    student_model = args[6]
    student_wd = args[7]
    alpha = args[8]
    temperature = args[9]
    

    """ check if initialized values exists """
    init_args = ['srm_init', dataset,  
            teacher_model, student_model, student_wd, 
            codeword_multiplier, sparsity_multiplier, 
            target_type]
    
    init_file = '_'.join([str(v) for v in init_args]) + '.pickle'
    init_file = os.path.join(Conf.OUTPUT_DIR, init_file)
    
    if os.path.exists(init_file):
        fid = open(init_file, 'rb')
        init_data = pickle.load(fid)
        fid.close()
    else:
        init_data = initialize_student(init_args)
        fid = open(init_file, 'wb')
        pickle.dump(init_data, fid)
        fid.close()
    
    
    test_mode = True if Conf.STAGE == 'test' else False
    
    if student_model == 'resnet18':
        batch_size = Conf.RESNET18_BS
        batch_size *= len(Conf.GPU_INDICES)
    else:
        raise RuntimeError('only support student resnet18')
     
    data = Datasets.get_data(dataset,
                             batch_size=batch_size,
                             test_mode=test_mode)

    train_loader, val_loader = data

    nb_class = get_class_count(dataset)

    if student_model == 'resnet18':
        student_getter = Models.resnet18
    else:
        raise RuntimeError('unsupported student')
   
    S_model = student_getter(pretrained=False)
    
    init_weights = init_data['model_weights']
    cur_weights = S_model.state_dict()
    
    for layer in cur_weights.keys():
        if layer in init_weights.keys():
            cur_weights[layer] = init_weights[layer]

    S_model.load_state_dict(cur_weights)

    if teacher_model == 'resnet34':
        teacher_getter = Models.resnet34
    else:
        raise RuntimeError('only support resnet34 as teacher')

    T_model = teacher_getter(pretrained=True)

    if len(Conf.GPU_INDICES) > 1:
        multigpu = True
        S_model = Layers.CustomParallel(S_model, device_ids=Conf.GPU_INDICES)
        T_model = Layers.CustomParallel(T_model, device_ids=Conf.GPU_INDICES)
    else:
        multigpu = False

    device = torch.device('cuda')
    S_model.to(device)
    T_model.to(device)
    
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
        verbose = True 
 

    outputs = Utility.knowledge_distil(logfile,
                                       S_model,
                                       T_model,
                                       compute,
                                       train_loader,
                                       val_loader,
                                       LR,
                                       Epochs,
                                       student_wd,
                                       alpha,
                                       temperature,
                                       verbose,
                                       multigpu)
   

        
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
    codeword_multiplier = args[3]
    sparsity_multiplier = args[4]

    assert dataset == 'imagenet', 'only support imagenet'

    """ load data """
    if teacher_model == 'resnet34':
        batch_size = Conf.CODEWORD34_BS
        batch_size *= len(Conf.GPU_INDICES)
    else:
        raise RuntimeError('only support resnet34 as teacher network')

    test_mode = True if Conf.STAGE == 'test' else False
    
    data = Datasets.get_data(dataset,
                             batch_size=batch_size,
                             test_mode=test_mode)

    train_loader, val_loader = data

    nb_class = get_class_count(dataset)

    compute = 'gpu' if torch.cuda.is_available() else 'cpu'

    """ check performance of teacher """
   
    assert teacher_model in ['resnet34']

    # note, pretrained weights have been loaded
    if teacher_model == 'resnet34':
        model = Models.sparse_rep_resnet34(codeword_multiplier, 
                                           sparsity_multiplier)
    else:
        raise RuntimeError('only support resnet34 as teacher network')

    if compute == 'gpu':
        device = torch.device('cuda')
        model.cuda(device)
    else:
        device = None

    if len(Conf.GPU_INDICES) > 1:
        model = Layers.CustomParallel(model, device_ids=Conf.GPU_INDICES)
        multigpu = True
    else:
        multigpu = False

    logfile = '_'.join([str(v) for v in args]) + '.pickle'
    logfile = os.path.join(Conf.LOG_DIR, logfile)

    if Conf.STAGE == 'test':
        LR = Conf.CLUSTER_TEST_LR
        Epochs = Conf.CLUSTER_TEST_EPOCH
        verbose = True
    else:
        LR = Conf.CLUSTER_LR
        Epochs = Conf.CLUSTER_EPOCH
        verbose = True 
 
    logdir = '_'.join([str(v) for v in args]) 
    logdir = os.path.join(Conf.LOG_DIR, logdir)
    if not os.path.exists(logdir):
        os.mkdir(logdir)

    outputs = Utility.train_codeword(logdir,
                                     model,
                                     device,
                                     train_loader,
                                     LR,
                                     Epochs,
                                     0.0,
                                     verbose,
                                     multigpu=multigpu)


        
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
    weight_decay = args[4]
    codeword_multiplier = args[5]
    sparsity_multiplier = args[6]
    target_type = args[7]


    """ check if codeword exists """

    codeword_args = ['codeword', dataset, 
                     teacher_model, codeword_multiplier,  
                     sparsity_multiplier]
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
        teacher_model = args[2]

    """ load data """
    test_mode = True if Conf.STAGE == 'test' else False
    
    if teacher_model == 'resnet34':
        batch_size = Conf.INIT34_BS
        batch_size *= len(Conf.GPU_INDICES)
    else:
        raise RuntimeError('only support resnet34 as teacher network')
    
    data = Datasets.get_data(dataset,
                             batch_size=batch_size,
                             test_mode=test_mode)

    train_loader, val_loader = data

    nb_class = get_class_count(dataset)

    compute = 'gpu' if torch.cuda.is_available() else 'cpu'
   

    """ build the model """
    assert target_type in ['global-local']

    assert student_model in ['resnet18']
    assert teacher_model in ['resnet34']
    
    student_model_getter = Models.resnet18_global_local_pred
    teacher_model_getter = Models.resnet34_global_local_label
      
 
    T_model = teacher_model_getter(codeword_multiplier, 
                                   sparsity_multiplier)
    
    S_model = student_model_getter(codeword_multiplier, T_model.in_dims)
    
    T_model.load_state_dict(codeword_data['model_weights'])

    if len(Conf.GPU_INDICES) > 1:
        multigpu = True
        S_model = Layers.CustomParallel(S_model, device_ids=Conf.GPU_INDICES)
        T_model = Layers.CustomParallel(T_model, device_ids=Conf.GPU_INDICES)
    else:
        multigpu = False
       
    device = torch.device('cuda')

    if Conf.STAGE == 'test':
        LR = Conf.INIT_TEST_LR
        Epochs = Conf.INIT_TEST_EPOCH
        verbose = True
    else:
        LR = Conf.INIT_LR
        Epochs = Conf.INIT_EPOCH
        verbose = True 
 

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
                               LR,
                               Epochs,
                               weight_decay,
                               verbose,
                               multigpu)

    return outputs



