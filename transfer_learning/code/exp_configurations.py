import os
import socket

GPU_INDICES = [0,]

DATA_DIR = '../../data/'

STAGE = 'deploy'

OUTPUT_DIR = os.path.join(os.path.dirname(os.getcwd()), 'output')
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

LOG_DIR = os.path.join(os.path.dirname(os.getcwd()), 'log')
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)


ALLCNN_SCALE = 1.0

RESNEXT50_BS = 240
ALLCNN_BS = 480
RESNET18_BS = 410
CW_RESNEXT50_BS = 120
HIST_ALLCNN_BS = 320
HIST_RESNET18_BS = 160 
DISTIL_ALLCNN_BS = 640 
DISTIL_RESNET18_BS = 430 

WORKERS = 8 if len(GPU_INDICES) == 1 else 40

T_LR = (1e-3, 1e-4, 1e-4, 1e-4, 1e-5)
T_EPOCH = (40, 40, 40, 40, 40)

T_TEST_LR = (1e-3, 1e-4)
T_TEST_EPOCH = (1, 1)


KD_LR = (1e-3, 1e-4, 1e-4, 1e-4, 1e-5)
KD_EPOCH = (40, 40, 40, 40, 40)

KD_TEST_LR = (1e-3, 1e-4)
KD_TEST_EPOCH = (1, 1)

S_LR =(1e-3, 1e-4, 1e-4, 1e-4, 1e-5) 
S_EPOCH = (40, 40, 40, 40, 40) 

S_TEST_LR = (1e-3, 1e-4)
S_TEST_EPOCH = (1, 1)

LINEAR_LR = (1e-3, 1e-4, 1e-5)
LINEAR_EPOCH = (10, 20, 10)

LINEAR_TEST_LR = (1e-3, 1e-4)
LINEAR_TEST_EPOCH = (1, 1)

CLUSTER_LR = (1e-3, 1e-4, 1e-4, 1e-4, 1e-5)
CLUSTER_EPOCH = (20, 30, 30, 30, 20)

CLUSTER_TEST_LR = (1e-4, 1e-4)
CLUSTER_TEST_EPOCH = (2, 2)

INIT_LR = (1e-3, 1e-4, 1e-4, 1e-5)
INIT_EPOCH = (20, 40, 40, 20)

INIT_TEST_LR = (1e-4, 1e-4)
INIT_TEST_EPOCH = (2, 2)

if STAGE != 'test':
    DATASETS = ['pubfig83',
                'pubfig83-5',
                'pubfig83-10',
                'cub',
                'cub-5',
                'cub-10',
                'flower102',
                'flower102-5',
                'mit_indoor',
                'mit_indoor-5',
                'mit_indoor-10',
                'stanford_cars',
                'stanford_cars-5',
                'stanford_cars-10']

    EXP = [0, 1, 2]
else:
    DATASETS = ['pubfig83', 'pubfig83-5', 'pubfig83-10']
    EXP = [0,]

# teacher hyperparameters 
teacher_names = ['prefix', 'teacher_model', 'dataset', 'weight_decay', 'exp']
teacher_values = {'prefix': ['teacher'],
                  'teacher_model': ['resnext50'],
                  'dataset': DATASETS,
                  'weight_decay': [1e-4,],
                  'exp': EXP}

# student hyperparameters
student_names = ['prefix', 'dataset', 'student_model', 
                 'weight_decay', 'exp']
student_values = {'prefix': ['student'],
                  'dataset': DATASETS,
                  'student_model': ['allcnn', 'resnet18'],
                  'weight_decay': [1e-4,],
                  'exp': EXP}

# KD hyperparameters
kd_names = ['prefix', 'dataset', 'teacher_model', 
            'student_model', 'teacher_wd', 'student_wd',
            'alpha', 'temperature', 'exp']

kd_values = {'prefix': ['kd'],
             'dataset': DATASETS,
             'teacher_model': ['resnext50'],
             'student_model': ['allcnn', 'resnet18'],
             'teacher_wd': [1e-4],
             'student_wd': [1e-4],
             'alpha': [0.5, 0.75],
             'temperature': [4.0, 8.0],
             'exp': EXP}

# FitNet initialization hyperparameters
hint_init_names = ['prefix', 'dataset', 'teacher_model', 
                   'student_model', 'teacher_wd', 'student_wd',
                    'exp']

hint_init_values = {'prefix': ['hint_init',],
                    'dataset': DATASETS,
                    'teacher_model': ['resnext50'],
                    'student_model': ['allcnn', 'resnet18'],
                    'teacher_wd': [1e-4,],
                    'student_wd': [1e-4],
                    'exp': EXP}

# FitNet hyperparameters
hintkd_names = ['prefix', 'dataset', 'teacher_model', 
                'student_model', 'teacher_wd', 'student_wd',
            'alpha', 'temperature', 'exp']

hintkd_values = {'prefix': ['hintkd'],
             'dataset': DATASETS,
             'teacher_model': ['resnext50'],
             'student_model': ['allcnn', 'resnet18'],
             'teacher_wd': [1e-4],
             'student_wd': [1e-4],
             'alpha': [0.5, 0.75],
             'temperature': [4.0, 8.0],
             'exp': EXP}

# SRM sparse representation hyperparameters
codeword_names = ['prefix', 'dataset', 'teacher_model', 'teacher_wd', 
                'codeword_multiplier', 'sparsity_multiplier', 
                'exp']

codeword_values = {'prefix': ['codeword'],
                 'dataset': DATASETS,
                 'teacher_model': ['resnext50'],
                 'teacher_wd': [1e-4,],
                 'codeword_multiplier': [2.0],
                 'sparsity_multiplier': [0.02],
                 'exp': EXP}

# SRM initialization hyperparameters
init_names = ['prefix', 'dataset', 'teacher_model', 'student_model', 
             'teacher_wd', 'student_wd', 'codeword_multiplier', 
             'sparsity_multiplier', 
              'target_type', 'exp']

init_values = {'prefix': ['srm_init'],
               'dataset': DATASETS,
               'teacher_model': ['resnext50'],
               'student_model': ['allcnn', 'resnet18'],
               'teacher_wd': [1e-4,],
               'student_wd': [1e-4,],
               'codeword_multiplier': [2.0],
               'sparsity_multiplier': [0.02],
               'target_type': ['global-local'],
               'exp': EXP}

# SRM KD hyperparameters
srmkd_names = ['prefix', 'dataset', 'teacher_model', 'student_model',  
            'codeword_multiplier', 'sparsity_multiplier', 
            'target_type', 'teacher_wd', 'student_wd',
            'alpha', 'temperature', 'exp']

srmkd_values = {'prefix': ['srmkd'],
             'dataset': DATASETS,
             'teacher_model': ['resnext50'],
             'student_model': ['allcnn', 'resnet18'],
             'codeword_multiplier': [2.0],
             'sparsity_multiplier': [0.02],
             'target_type': ['global-local'],
             'teacher_wd': [1e-4],
             'student_wd': [1e-4],
             'alpha': [0.5, 0.75],
             'temperature': [4.0, 8.0],
             'exp': EXP}


