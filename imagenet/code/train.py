#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, getopt
import JobScheduler
import exp_configurations as Conf
import Runners
import copy
import itertools
import pickle

def create_configuration(names, values):
    outputs_ = []

    tmp = []
    for name in names:
        tmp.append(values[name])
    outputs_ += list(itertools.product(*tmp))

    outputs = []
    for conf in outputs_:
        if conf not in outputs:
            outputs.append(conf)
    return outputs

def train(names, values, executor):
    configurations = create_configuration(names, values)

    for conf in configurations:
        filename = '_'.join([str(v) for v in conf]) + '.pickle'
        filename = os.path.join(Conf.OUTPUT_DIR, filename)
        if not os.path.exists(filename):
            outputs = executor(conf)
            fid = open(filename, 'wb')
            pickle.dump(outputs, fid)
            fid.close()

def main(argv):
    
    # train sparse representation
    print('extract sparse representation')
    train(Conf.codeword_names, Conf.codeword_values, Runners.train_codeword)
    # initialize using SRM
    print('initialize student using SRM')
    train(Conf.init_names, Conf.init_values, Runners.initialize_student)
    # train student using SRM
    print('train student using SRM')
    train(Conf.srmkd_names, Conf.srmkd_values, Runners.SRMKD)
    
   
    
if __name__ == "__main__":
    main(sys.argv[1:])
