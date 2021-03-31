import os



OUTPUT_DIR = 'output'
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

BATCHJOB_DIR = 'batchjob'
if not os.path.exists(BATCHJOB_DIR):
    os.mkdir(BATCHJOB_DIR)

TMP_DIR = 'tmp'
if not os.path.exists(TMP_DIR):
    os.mkdir(TMP_DIR)

mode = 'deploy'

exp = [1, 2, 3] if mode == 'deploy' else [1]

if mode == 'deploy':
    tl_datasets = ['flower102', 'CUB', 'stanford_cars', 'mit_indoor', 'pubfig83',
                        'flower102-5', 'CUB-5', 'stanford_cars-5', 'mit_indoor-5', 'pubfig83-5',
                        'CUB-10', 'stanford_cars-10', 'mit_indoor-10', 'pubfig83-10']
else:
    tl_datasets = ['flower102', 'flower102-5']

method_conf = {'attention': {'r': 1,
                             'a': 0,
                             'b': 1000},
               'rkd': {'r': 1,
                       'a': 0,
                       'b': 1},
               'pkt': {'r': 1,
                       'a': 0,
                       'b': 30000},
               'crd': {'r': 1,
                       'a': 1,
                       'b': 0.8}}



names = ['dataset', 'method', 'exp', 'student']
values_cifar100 = {'dataset': ['cifar100'],
          'method': ['attention', 'crd', 
                     'rkd', 'pkt'],

          'exp': exp,
          'student': ['allcnn']}

values_tl = {'dataset': tl_datasets,
          'method': ['attention', 'crd',
                     'rkd', 'pkt'],

          'exp': exp,
          'student': ['resnet18', 'allcnnt']}

