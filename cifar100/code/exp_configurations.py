import os
import socket

DATA_DIR = '../../data/'

STAGE = 'deploy'

OUTPUT_DIR = os.path.join(os.path.dirname(os.getcwd()), 'output')
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

LOG_DIR = os.path.join(os.path.dirname(os.getcwd()), 'log')
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)



DENSENET_BS = 128
ALLCNN_BS = 256
CODEWORD_BS = 512 
DISTIL_BS = 128
HISTOGRAM_BS = 384
COUNT_BS = 128 

   
WORKERS = 4

T_LR = (1e-3, 1e-4, 1e-5)
T_EPOCH = (30, 100, 70)

T_TEST_LR = (1e-3, 1e-4)
T_TEST_EPOCH = (1, 1)

KD_LR = (1e-3, 1e-4, 1e-5)
KD_EPOCH = (30, 100, 70)

KD_TEST_LR = (1e-3, 1e-4)
KD_TEST_EPOCH = (1, 1)


S_LR = (1e-3, 1e-4, 1e-5)
S_EPOCH = (30, 100, 70)

S_TEST_LR = (1e-3, 1e-4)
S_TEST_EPOCH = (1, 1)

LINEAR_LR = (1e-3, 1e-4, 1e-5)
LINEAR_EPOCH = (10, 20, 10)

LINEAR_TEST_LR = (1e-3, 1e-4)
LINEAR_TEST_EPOCH = (1, 1)


CLUSTER_LR = (1e-3, 1e-4, 1e-5)
CLUSTER_EPOCH = (10, 60, 30)

CLUSTER_TEST_LR = (1e-4, 1e-4)
CLUSTER_TEST_EPOCH = (2, 2)

INIT_LR = (1e-3, 1e-4, 1e-5)
INIT_EPOCH = (30, 100, 30)

INIT_TEST_LR = (1e-4, 1e-4)
INIT_TEST_EPOCH = (2, 2)

if STAGE == 'test':
    EXP = [0, ]
else:
    EXP = [0, 1, 2]

teacher_names = ['prefix', 'dataset', 'weight_decay', 'exp']
teacher_values = {'prefix': ['teacher'],
                  'dataset': ['cifar100'],
                  'weight_decay': [1e-4,],
                  'exp': EXP}

student_names = ['prefix', 'dataset', 'scale', 'weight_decay', 'exp']
student_values = {'prefix': ['student'],
                  'dataset': ['cifar100'],
                  'scale': [0.66],
                  'weight_decay': [1e-4,],
                  'exp': EXP}


kd_names = ['prefix', 'dataset', 'scale', 'teacher_wd', 'student_wd',
            'alpha', 'temperature', 'exp']

kd_values = {'prefix': ['kd'],
             'dataset': ['cifar100',],
             'scale': [0.66,],
             'teacher_wd': [1e-4],
             'student_wd': [1e-4],
             'alpha': [0.25, 0.5, 0.75],
             'temperature': [2.0, 4.0, 8.0],
             'exp': EXP}

# parameters for initializing FitNet
hint_init_names = ['prefix', 'dataset', 'scale', 'teacher_wd', 'student_wd',
                    'exp']

hint_init_values = {'prefix': ['hint_init',],
                    'dataset': ['cifar100'],
                    'scale': [0.66],
                    'teacher_wd': [1e-4,],
                    'student_wd': [1e-4],
                    'exp': EXP}

# parameters of FitNet
hintkd_names = ['prefix', 'dataset', 'scale', 'teacher_wd', 'student_wd',
            'alpha', 'temperature', 'exp']

hintkd_values = {'prefix': ['hintkd'],
             'dataset': ['cifar100',],
             'scale': [0.66,],
             'teacher_wd': [1e-4],
             'student_wd': [1e-4],
             'alpha': [0.25, 0.5, 0.75],
             'temperature': [2.0, 4.0, 8.0],
             'exp': EXP}

# linear probing + whole network update of FitNet after initialization
hintkdwt_names = ['prefix', 'dataset', 'scale', 'teacher_wd', 'student_wd',
            'exp']

hintkdwt_values = {'prefix': ['hintkdwt'],
             'dataset': ['cifar100',],
             'scale': [0.66,],
             'teacher_wd': [1e-4],
             'student_wd': [1e-4],
             'exp': EXP}

# sparse representation learning of teacher
codeword_names = ['prefix', 'dataset', 'teacher_wd', 
                'codeword_multiplier', 'sparsity_multiplier', 
                'exp']

codeword_values = {'prefix': ['codeword'],
                 'dataset': ['cifar100'],
                 'teacher_wd': [1e-4,],
                 'codeword_multiplier': [1.5, 2.0, 3.0],
                 'sparsity_multiplier': [0.01, 0.02, 0.03],
                 'exp': EXP}

# initializing student using SRM
init_names = ['prefix', 'dataset', 'teacher_wd',
              'student_wd', 'codeword_multiplier', 'sparsity_multiplier', 
              'scale', 'target_type', 'exp']

init_values = {'prefix': ['srm_init'],
               'dataset': ['cifar100'],
               'teacher_wd': [1e-4,],
               'student_wd': [1e-4,],
               'codeword_multiplier': [1.5, 2.0, 3.0],
               'sparsity_multiplier': [0.01, 0.02, 0.03],
               'scale': [0.66,],
               'target_type': ['local', 'global', 'global-local'],
               'exp': EXP}

# KD with SRM
srmkd_names = ['prefix', 'dataset', 'scale',
            'codeword_multiplier', 'sparsity_multiplier', 
            'init_type', 'teacher_wd', 'student_wd',
            'alpha', 'temperature', 'exp']

srmkd_values = {'prefix': ['srmkd'],
             'dataset': ['cifar100',],
             'scale': [0.66,],
             'codeword_multiplier': [1.5, 2.0, 3.0],
             'sparsity_multiplier': [0.01, 0.02, 0.03],
             'init_type': ['global-local', 'local', 'global'],
             'teacher_wd': [1e-4],
             'student_wd': [1e-4],
             'alpha': [0.25, 0.5, 0.75],
             'temperature': [2.0, 4.0, 8.0],
             'exp': EXP}

# linear probing + whole network update after initialization with SRM
srmkdwt_names = ['prefix', 'dataset', 'scale',
            'codeword_multiplier', 'sparsity_multiplier',  
            'init_type', 'teacher_wd', 'student_wd',
            'exp']

srmkdwt_values = {'prefix': ['srmkdwt'],
             'dataset': ['cifar100',],
             'scale': [0.66,],
             'codeword_multiplier': [2.0,],
             'sparsity_multiplier': [0.02,],
             'init_type': ['global-local'],
             'teacher_wd': [1e-4],
             'student_wd': [1e-4],
             'exp': EXP}

