import os
import socket

GPU_INDICES = [0, 1]


DATA_DIR = '../../data/'

STAGE = 'deploy'


OUTPUT_DIR = os.path.join(os.path.dirname(os.getcwd()), 'output')
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

LOG_DIR = os.path.join(os.path.dirname(os.getcwd()), 'log')
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)



DENSENET_BS = 320 
RESNET18_BS = 640 
CODEWORD34_BS = 320 
CODEWORD50_BS = 160 
INIT34_BS = 320 
INIT50_BS = 240 
DISTIL_BS = 320 


    
WORKERS = 20 if len(GPU_INDICES) == 1 else 40

T_LR = (1e-3, 1e-4, 1e-5)
T_EPOCH = (30, 100, 70)

T_TEST_LR = (1e-3, 1e-4)
T_TEST_EPOCH = (1, 1)


KD_LR = [1e-1,]*50 + [1e-2]*30 + [1e-3,]*10 + [1e-4,]*10 
KD_EPOCH = [1,]*50 + [1,]*30 + [1,]*10 + [1,]*10

KD_TEST_LR = (1e-3, 1e-4)
KD_TEST_EPOCH = (1, 1)


S_LR = (1e-3, 1e-4, 1e-5)
S_EPOCH = (30, 100, 70)

S_TEST_LR = (1e-3, 1e-4)
S_TEST_EPOCH = (1, 1)

CLUSTER_LR = [1e-3]*15 + [1e-4,]*10 + [5e-5]*10 
CLUSTER_EPOCH = [1,]*15 + [1,]*10 + [1,]*10

CLUSTER_TEST_LR = (1e-2, 1e-3, 1e-4)
CLUSTER_TEST_EPOCH = (1, 1)

INIT_LR = [1e-1,]*20 + [1e-2]*20 + [1e-3,]*20 + [1e-4,]*20 
INIT_EPOCH = [1,]*20 + [1,]*20 + [1,]*20 + [1,]*20 

INIT_TEST_LR = (1e-1, 1e-2)
INIT_TEST_EPOCH = (1, 1)

# SRM sparse representation hyperparameters
codeword_names = ['prefix', 'dataset', 'teacher_model', 
                'codeword_multiplier', 'sparsity_multiplier']

codeword_values = {'prefix': ['codeword'],
                 'dataset': ['imagenet'],
                 'teacher_model': ['resnet34'],
                 'codeword_multiplier': [4.0],
                 'sparsity_multiplier': [0.02]}

# SRM initialization hyperparameters
init_names = ['prefix', 'dataset', 'teacher_model', 'student_model',
              'weight_decay', 'codeword_multiplier', 
              'sparsity_multiplier', 
              'target_type']

init_values = {'prefix': ['srm_init'],
               'dataset': ['imagenet'],
               'teacher_model': ['resnet34',],
               'student_model': ['resnet18'],
               'weight_decay': [1e-4,],
               'codeword_multiplier': [4.0],
               'sparsity_multiplier': [0.02],
               'target_type': ['global-local']}

# SRM KD hyperparameters
srmkd_names = ['prefix', 'dataset', 
            'codeword_multiplier', 'sparsity_multiplier', 
            'target_type', 'teacher_model', 
            'student_model', 'weight_decay',
            'alpha', 'temperature']

srmkd_values = {'prefix': ['srmkd'],
             'dataset': ['imagenet',],
             'codeword_multiplier': [4.0,],
             'sparsity_multiplier': [0.02,],
             'target_type': ['global-local'],
             'teacher_model': ['resnet34'],
             'student_model': ['resnet18'],
             'weight_decay': [1e-4],
             'alpha': [0.3],
             'temperature': [4.0,]}


