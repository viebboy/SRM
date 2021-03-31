import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss
import pickle
import os
from tqdm import tqdm
import numpy as np
import Layers
import torch.nn as nn
import torch.nn.functional as F
import exp_configurations as Conf



def KD_loss(outputs, 
               labels, 
               teacher_outputs, 
               alpha, 
               temperature):
   
    T = temperature
    loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs/temperature, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels) * (1. - alpha)

    return loss



def initialize_hint(logdir,
                    S_model,
                    T_model,
                    compute,
                    train_loader,
                    LR,
                    Epochs,
                    weight_decay,
                    verbose):

    S_model = S_model.float()
    T_model = T_model.float()

    logfile = os.path.join(logdir, 'current_state.pickle')
    model_file = os.path.join(logdir, 'best_model.pickle')

    if os.path.exists(logfile):
        fid = open(logfile, 'rb')
        data = pickle.load(fid)
        fid.close()

        S_model.load_state_dict(data['model_state'])
        LR = data['LR']
        Epoch = data['Epochs']
        epoch_offset = data['epoch_offset']
        losses = data['losses']

    else:
        losses = []
        epoch_offset = 0

    if compute == 'gpu':
        device = torch.device('cuda')
        S_model.to(device)
        T_model.to(device)
        torch.backends.cudnn.benchmark = True
    else:
        device = None


    for idx, (lr, epochs) in enumerate(zip(LR, Epochs)):
        losses_ = hint_epoch_trainer(S_model,
                                      T_model, 
                                      train_loader,
                                      device,
                                      lr,
                                      epochs,
                                      epoch_offset,
                                      weight_decay,
                                      verbose)

        losses += losses_

        epoch_offset += epochs

        # log data
        weights = S_model.cpu().state_dict() 

        data = {'model_state': weights,
                'LR': LR[idx+1:],
                'Epochs': Epochs[idx+1:],
                'epoch_offset': epoch_offset,
                'losses': losses}

        fid = open(logfile, 'wb')
        pickle.dump(data, fid)
        fid.close()
   
        if device is not None: 
            S_model.to(device)
   
    S_model.cpu()
    outputs = {'model_weights': S_model.state_dict(),
               'train_loss': losses}

    fid = open(model_file, 'wb')
    pickle.dump({'model_weights': S_model.state_dict()}, fid)
    fid.close()

    return outputs


def hint_epoch_trainer(S_model,
                       T_model,
                       data_loader,
                       device,
                       lr=0.0001,
                       epochs=1,
                       epoch_offset=0,
                       weight_decay=1e-4,
                       verbose=0):

    optimizer = optim.Adam
    
    S_model.train()
    T_model.eval()
    parameters = S_model.parameters()
    model_optimizer = optimizer(parameters, lr=lr, weight_decay=weight_decay)

    losses = []
    accuracy = []

    L = MSELoss()
    
    for epoch_idx in range(epochs):
        if verbose:
            loader = tqdm(data_loader,
                    desc='epoch ' + str(epoch_offset+epoch_idx+1)+': ',
                          ncols=80,
                          ascii=True)
        else:
            loader = data_loader

        train_loss, counter =  0, 0
        
        for inputs, targets in loader:
            model_optimizer.zero_grad()
            
            if device is not None:
                inputs = inputs.cuda(device)
                targets = targets.cuda(device)

            s_hint1, s_hint2, s_hint3 = S_model(inputs)
            with torch.no_grad():
                t_hint1, t_hint2, t_hint3 = T_model(inputs)

            loss = L(s_hint1, t_hint1) + L(s_hint2, t_hint2) +\
                    L(s_hint3, t_hint3)

           
            loss.backward()
            model_optimizer.step()

            train_loss += loss.item()
            counter += targets.size(0)

        losses.append(train_loss / counter)
        print('epoch: %s' % str(epoch_offset + epoch_idx + 1))
        print('--loss: %.6f' % (train_loss / counter))

    return losses

def kd_epoch_trainer(S_model,
                    T_model,
                    train_loader,
                    test_loader,
                    device,
                    lr=0.0001,
                    epochs=1,
                    epoch_offset=0,
                    weight_decay=1e-4,
                    alpha=0.5,
                    temperature=1.0,
                    verbose=0):

    optimizer = optim.Adam
    
    S_model.train()
    T_model.eval()
    parameters = S_model.parameters()
    model_optimizer = optimizer(parameters, lr=lr, weight_decay=weight_decay)

    train_loss = []
    train_accuracy = []
    test_loss = []
    test_accuracy = []

    
    for epoch_idx in range(epochs):
        if verbose:
            loader = tqdm(train_loader,
                    desc='epoch ' + str(epoch_offset+epoch_idx+1)+': ',
                          ncols=80,
                          ascii=True)
        else:
            loader = train_loader

        for inputs, targets in loader:
            model_optimizer.zero_grad()
            
            if device is not None:
                inputs = inputs.cuda(device)
                targets = targets.cuda(device)

            outputs = S_model(inputs)
            with torch.no_grad():
                teacher_outputs = T_model(inputs)

            targets = torch.squeeze(targets)
            loss = KD_loss(outputs, 
                           targets, 
                           teacher_outputs, 
                           alpha, 
                           temperature)
            
           
            loss.backward()
            model_optimizer.step()


        train_outputs = classifier_evaluator(S_model, train_loader, device)
        test_outputs = classifier_evaluator(S_model, test_loader, device)

        train_loss.append(train_outputs[0])
        train_accuracy.append(train_outputs[1])
        test_loss.append(test_outputs[0])
        test_accuracy.append(test_outputs[1])

        S_model.train()

    return train_loss, train_accuracy, test_loss, test_accuracy 



def knowledge_distil(logfile,
                    S_model,
                    T_model,
                    compute,
                    train_loader,
                    val_loader,
                    test_loader,
                    LR,
                    Epochs,
                    weight_decay,
                    alpha,
                    temperature,
                    verbose):

    model_file = logfile[:-7] + '_best.pickle'

    S_model = S_model.float()
    T_model = T_model.float()

    if os.path.exists(logfile):
        fid = open(logfile, 'rb')
        data = pickle.load(fid)
        fid.close()

        S_model.load_state_dict(data['model_state'])
        LR = data['LR']
        Epoch = data['Epochs']
        epoch_offset = data['epoch_offset']
        train_loss = data['train_loss']
        train_accuracy = data['train_accuracy']
        test_loss = data['test_loss']
        test_accuracy = data['test_accuracy']
        val_measure = data['val_measure']

    else:
        train_loss = []
        train_accuracy = []
        test_loss = []
        test_accuracy = []
        epoch_offset = 0
        val_measure = 0

    if compute == 'gpu':
        device = torch.device('cuda')
        S_model.cuda(device)
        T_model.cuda(device)
        torch.backends.cudnn.benchmark = True
    else:
        device = None


    for idx, (lr, epochs) in enumerate(zip(LR, Epochs)):
        epoch_outputs = kd_epoch_trainer(S_model,
                                         T_model, 
                                         train_loader,
                                         test_loader,
                                         device,
                                         lr,
                                         epochs,
                                         epoch_offset,
                                         weight_decay,
                                         alpha,
                                         temperature,
                                         verbose)

        train_loss_, train_accuracy_, test_loss_, test_accuracy_ = epoch_outputs
        train_loss += train_loss_
        train_accuracy += train_accuracy_
        test_loss += test_loss_
        test_accuracy += test_accuracy_

        epoch_offset += epochs

        if val_loader is not None:
            # evaluate on train set
            _, val_acc = classifier_evaluator(S_model, 
                                              val_loader, 
                                              device, 
                                              verbose)
            print('val acc: %.2f' % val_acc)

            if val_acc > val_measure:
                val_measure = val_acc
                if device is not None:
                    best_model_state = S_model.cpu().state_dict() 
                else:
                    best_model_state = S_model.state_dict()
        else:
            if device is not None:
                best_model_state = S_model.cpu().state_dict()
            else:
                best_model_state = S_model.state_dict()

        fid = open(model_file, 'wb')
        pickle.dump(best_model_state, fid)
        fid.close()

        
        
        # log data
        if device is not None:
            weights = S_model.cpu().state_dict()
        else:
            weights = S_model.state_dict()

        data = {'model_state': weights,
                'LR': LR[idx+1:],
                'Epochs': Epochs[idx+1:],
                'epoch_offset': epoch_offset,
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                'test_loss': test_loss,
                'test_accuracy': test_accuracy,
                'val_measure': val_measure}

        fid = open(logfile, 'wb')
        pickle.dump(data, fid)
        fid.close()

        if device is not None:
            S_model.to(device)

    fid = open(model_file, 'rb')
    best_model_state = pickle.load(fid)
    fid.close()

    if device is not None:
        S_model.cpu()
        S_model.load_state_dict(best_model_state)
        S_model.to(device)
    else:
        S_model.load_state_dict(best_model_state)

    _, train_acc = classifier_evaluator(S_model, train_loader, device, verbose)
    _, test_acc = classifier_evaluator(S_model, test_loader, device, verbose)

    if val_loader is not None:
        _, val_acc = classifier_evaluator(S_model, val_loader, device, verbose)
    else:
        val_acc = 0

    print('train acc: %.4f' % train_acc)
    print('val acc: %.4f' % val_acc)
    print('test acc: %.4f' % test_acc)

    outputs = {'model_weights': S_model.state_dict(),
               'train_acc': train_acc,
               'val_acc': val_acc,
               'test_acc': test_acc,
               'train_loss': train_loss,
               'train_accuracy': train_accuracy,
               'test_loss': test_loss,
               'test_accuracy': test_accuracy}

    return outputs


def train_codeword(logdir,
                   model,
                   compute,
                   data_loader,
                   LR,
                   Epochs,
                   weight_decay,
                   verbose):

    model_file = os.path.join(logdir, 'best_model.pickle')
    logfile = os.path.join(logdir, 'current_state.pickle')

    model = model.float()
    
    torch.backends.cudnn.benchmark = True   
    
    if os.path.exists(logfile):
        fid = open(logfile, 'rb')
        data = pickle.load(fid)
        fid.close()

        model.load_state_dict(data['model_state'])
        LR = data['LR']
        Epoch = data['Epochs']
        epoch_offset = data['epoch_offset']
        losses = data['losses']
        val_measure = data['val_measure']

    else:
        losses = []
        epoch_offset = 0
        val_measure = 1e10

    if compute == 'gpu':
        device = torch.device('cuda')
        model.to(device)
        torch.backends.cudnn.benchmark = True
    else:
        device = None


    for idx, (lr, epochs) in enumerate(zip(LR, Epochs)):
        losses_  = codeword_epoch_trainer(model,
                                          data_loader,
                                          device,
                                          lr,
                                          epochs,
                                          epoch_offset,
                                          weight_decay,
                                          verbose)

        losses += losses_

        epoch_offset += epochs

        # if train loss improve, save the weights
        if losses[-1] < val_measure: 
            val_measure = losses[-1]
            best_model_state = model.cpu().state_dict()
            fid = open(model_file, 'wb')
            pickle.dump({'model_weights':best_model_state}, fid)
            fid.close()
            if device is not None: 
                model.to(device)

        # log data
        weights = model.cpu().state_dict()
        
        data = {'model_state': weights,
                'LR': LR[idx+1:],
                'Epochs': Epochs[idx+1:],
                'epoch_offset': epoch_offset,
                'losses': losses,
                'val_measure': val_measure}

        fid = open(logfile, 'wb')
        pickle.dump(data, fid)
        fid.close()
        if device is not None: 
            model.to(device)
    
    fid = open(model_file, 'rb')
    best_model_state = pickle.load(fid)['model_weights']
    fid.close()

    model.cpu()
    model.load_state_dict(best_model_state)

    outputs = {'model_weights': model.state_dict(),
               'train_loss': losses}

    return outputs

def srm_init(logdir,
            target_type,
            S_model,
            T_model,
            compute,
            train_loader,
            val_loader,
            test_loader,
            LR,
            Epochs,
            weight_decay,
            verbose):
    
    model_file = os.path.join(logdir, 'best_model.pickle')
    logfile = os.path.join(logdir, 'current_state.pickle')

    S_model = S_model.float()
    T_model = T_model.float()

    if os.path.exists(logfile):
        fid = open(logfile, 'rb')
        data = pickle.load(fid)
        fid.close()

        S_model.load_state_dict(data['model_state'])
        LR = data['LR']
        Epoch = data['Epochs']
        epoch_offset = data['epoch_offset']
        losses = data['losses']
        val_measure = data['val_measure']

    else:
        losses = []
        epoch_offset = 0
        val_measure = 1e10

    if compute == 'gpu':
        device = torch.device('cuda')
        S_model.to(device)
        T_model.to(device)

        torch.backends.cudnn.benchmark = True
    else:
        device = None


    for idx, (lr, epochs) in enumerate(zip(LR, Epochs)):
        losses_ = srm_init_epoch_trainer(target_type,
                                        S_model,
                                        T_model,
                                        train_loader,
                                        device,
                                        lr,
                                        epochs,
                                        epoch_offset,
                                        weight_decay,
                                        verbose)

        losses += losses_

        epoch_offset += epochs

        if val_loader is not None:
            # evaluate on train set
            val_loss = srm_init_evaluator(target_type, 
                                          S_model, 
                                          T_model, 
                                          val_loader, 
                                          device, 
                                          verbose)
            print('val loss: %.5f' % val_loss)

            if val_loss < val_measure:
                val_measure = val_loss
                best_model_state = S_model.cpu().state_dict()
                fid = open(model_file, 'wb')
                pickle.dump({'model_weights': best_model_state}, fid)
                fid.close()
        else:
            best_model_state = S_model.cpu().state_dict()
            fid = open(model_file, 'wb')
            pickle.dump({'model_weights':best_model_state}, fid)
            fid.close()

        # log data
        weights = S_model.cpu().state_dict()

        data = {'model_state': weights,
                'LR': LR[idx+1:],
                'Epochs': Epochs[idx+1:],
                'epoch_offset': epoch_offset,
                'losses': losses,
                'val_measure': val_measure}

        fid = open(logfile, 'wb')
        pickle.dump(data, fid)
        fid.close()
        if device is not None: 
            S_model.to(device)
    
    fid = open(model_file, 'rb')
    best_model_state = pickle.load(fid)['model_weights']
    fid.close()

    S_model.cpu()
    S_model.load_state_dict(best_model_state)
    if device is not None: 
        S_model.to(device)

    train_loss = srm_init_evaluator(target_type, 
                                    S_model, 
                                    T_model, 
                                    train_loader, 
                                    device, 
                                    verbose)
    test_loss = srm_init_evaluator(target_type, 
                                   S_model, 
                                   T_model, 
                                   test_loader, 
                                   device, 
                                   verbose)

    if val_loader is not None:
        val_loss = srm_init_evaluator(target_type, 
                                      S_model, 
                                      T_model, 
                                      val_loader, 
                                      device, 
                                      verbose)
    else:
        val_loss = 0

    print('train loss: %.6f' % train_loss)
    print('val loss: %.6f' % val_loss)
    print('test loss: %.6f' % test_loss)

    outputs = {'model_weights': S_model.cpu().state_dict(),
               'train_loss': train_loss,
               'val_loss': val_loss,
               'test_loss': test_loss,
               'train_losses': losses}

    return outputs


def train_linear_classifier(logfile,
                            model,
                            compute,
                            train_loader,
                            val_loader,
                            test_loader,
                            LR,
                            Epochs,
                            weight_decay,
                            verbose):

    model_file = logfile[:-7] + '_best.pickle'

    model = model.float()
    if os.path.exists(logfile):
        fid = open(logfile, 'rb')
        data = pickle.load(fid)
        fid.close()

        model.load_state_dict(data['model_state'])
        LR = data['LR']
        Epoch = data['Epochs']
        epoch_offset = data['epoch_offset']
        losses = data['losses']
        accuracy = data['accuracy']
        val_measure = data['val_measure']

    else:
        losses = []
        accuracy = []
        epoch_offset = 0
        val_measure = 0

    if compute == 'gpu':
        device = torch.device('cuda')
        model.cuda(device)
        torch.backends.cudnn.benchmark = True
    else:
        device = None


    for idx, (lr, epochs) in enumerate(zip(LR, Epochs)):
        losses_, accuracy_ = linear_classifier_epoch_trainer(model,
                                                            train_loader,
                                                            device,
                                                            lr,
                                                            epochs,
                                                            epoch_offset,
                                                            weight_decay,
                                                            verbose)

        losses += losses_
        accuracy += accuracy_

        epoch_offset += epochs

        if val_loader is not None:
            # evaluate on train set
            _, val_acc = classifier_evaluator(model, 
                                           val_loader, 
                                           device, 
                                           verbose)
            print('val acc: %.2f' % val_acc)

            if val_acc > val_measure:
                print('saving best model')
                val_measure = val_acc
                if device is not None:
                    best_model_state = model.cpu().state_dict()
                    fid = open(model_file, 'wb')
                    pickle.dump(best_model_state, fid)
                    fid.close()
                else:
                    best_model_state = model.state_dict()
                    fid = open(model_file, 'wb')
                    pickle.dump(best_model_state, fid)
                    fid.close()
        else:
            if device is not None:
                best_model_state = model.cpu().state_dict()
                fid = open(model_file, 'wb')
                pickle.dump(best_model_state, fid)
                fid.close()
            else:
                best_model_state = model.state_dict()
                fid = open(model_file, 'wb')
                pickle.dump(best_model_state, fid)
                fid.close()
        
        weights = model.cpu().state_dict()

        data = {'model_state': weights,
                'LR': LR[idx+1:],
                'Epochs': Epochs[idx+1:],
                'epoch_offset': epoch_offset,
                'losses': losses,
                'accuracy': accuracy,
                'val_measure': val_measure}

        fid = open(logfile, 'wb')
        pickle.dump(data, fid)
        fid.close()
        if device is not None: 
            model.to(device)

    fid = open(model_file, 'rb')
    best_model_state = pickle.load(fid)
    fid.close()

    if device is not None:
        model.cpu()
        model.load_state_dict(best_model_state)
        model.to(device)
    else:
        model.load_state_dict(best_model_state)

    _, train_acc = classifier_evaluator(model, 
                                        train_loader, 
                                        device, 
                                        verbose)
    _, test_acc = classifier_evaluator(model, 
                                       test_loader, 
                                       device, 
                                       verbose)

    if val_loader is not None:
        _, val_acc = classifier_evaluator(model, 
                                          val_loader, 
                                          device, 
                                          verbose)
    else:
        val_acc = 0

    print('train acc: %.2f' % train_acc)
    print('val acc: %.2f' % val_acc)
    print('test acc: %.2f' % test_acc)

    outputs = {'model_weights': model.state_dict(),
               'train_acc': train_acc,
               'val_acc': val_acc,
               'test_acc': test_acc,
               'train_loss': losses,
               'train_accuracy': accuracy}

    return outputs


def train_classifier(logfile,
                    model,
                    compute,
                    train_loader,
                    val_loader,
                    test_loader,
                    LR,
                    Epochs,
                    weight_decay,
                    verbose):

    model_file = logfile[:-7] + '_best.pickle'

    model = model.float()
    if os.path.exists(logfile):
        fid = open(logfile, 'rb')
        data = pickle.load(fid)
        fid.close()

        model.load_state_dict(data['model_state'])
        LR = data['LR']
        Epoch = data['Epochs']
        epoch_offset = data['epoch_offset']
        losses = data['losses']
        accuracy = data['accuracy']
        val_measure = data['val_measure']

    else:
        losses = []
        accuracy = []
        epoch_offset = 0
        val_measure = 0

    if compute == 'gpu':
        device = torch.device('cuda')
        model.cuda(device)
        torch.backends.cudnn.benchmark = True
    else:
        device = None


    for idx, (lr, epochs) in enumerate(zip(LR, Epochs)):
        losses_, accuracy_ = classifier_epoch_trainer(model,
                                                    train_loader,
                                                    device,
                                                    lr,
                                                    epochs,
                                                    epoch_offset,
                                                    weight_decay,
                                                    verbose)

        losses += losses_
        accuracy += accuracy_

        epoch_offset += epochs

        if val_loader is not None:
            # evaluate on train set
            _, val_acc = classifier_evaluator(model, 
                                              val_loader, 
                                              device, 
                                              verbose)
            print('val acc: %.2f' % val_acc)

            if val_acc > val_measure:
                val_measure = val_acc
                if device is not None:
                    best_model_state = model.cpu().state_dict()
                    fid = open(model_file, 'wb')
                    pickle.dump(best_model_state, fid)
                    fid.close()
                else:
                    best_model_state = model.state_dict()
                    fid = open(model_file, 'wb')
                    pickle.dump(best_model_state, fid)
                    fid.close()

        else:
            if device is not None:
                best_model_state = model.cpu().state_dict()
                fid = open(model_file, 'wb')
                pickle.dump(best_model_state, fid)
                fid.close()
            else:
                best_model_state = model.state_dict()
                fid = open(model_file, 'wb')
                pickle.dump(best_model_state, fid)
                fid.close()


        # log data
        weights = model.cpu().state_dict()

        data = {'model_state': weights,
                'LR': LR[idx+1:],
                'Epochs': Epochs[idx+1:],
                'epoch_offset': epoch_offset,
                'losses': losses,
                'accuracy': accuracy,
                'val_measure': val_measure}

        fid = open(logfile, 'wb')
        pickle.dump(data, fid)
        fid.close()
        if device is not None:
            model.to(device)

    fid = open(model_file, 'rb')
    best_model_state = pickle.load(fid)
    fid.close()

    if device is not None:
        model.cpu()
        model.load_state_dict(best_model_state)
        model.to(device)
    else:
        model.load_state_dict(best_model_state)

    _, train_acc = classifier_evaluator(model, 
                                        train_loader, 
                                        device, 
                                        verbose)
    _, test_acc = classifier_evaluator(model, 
                                       test_loader, 
                                       device, 
                                       verbose)

    if val_loader is not None:
        _, val_acc = classifier_evaluator(model, 
                                          val_loader, 
                                          device, 
                                          verbose)
    else:
        val_acc = 0

    print('train acc: %.2f' % train_acc)
    print('val acc: %.2f' % val_acc)
    print('test acc: %.2f' % test_acc)

    outputs = {'model_weights': model.state_dict(),
               'train_acc': train_acc,
               'val_acc': val_acc,
               'test_acc': test_acc,
               'train_loss': losses,
               'train_accuracy': accuracy}

    return outputs

def codeword_epoch_trainer(model,
                           data_loader,
                           device,
                           lr=0.0001,
                           epochs=1,
                           epoch_offset=0,
                           weight_decay=1e-4,
                           verbose=0):

    optimizer = optim.Adam
    
    model.train()
    
    parameters = model.sparse_layers.parameters()
    model_optimizer = optimizer(parameters, lr=lr)

    losses = []

    L = MSELoss()
    
    for epoch_idx in range(epochs):
        if verbose:
            loader = tqdm(data_loader,
                    desc='epoch ' + str(epoch_offset+epoch_idx+1)+': ',
                          ncols=80,
                          ascii=True)
        else:
            loader = data_loader

        train_loss, counter = 0, 0
        
        for inputs, _ in loader:
            model_optimizer.zero_grad()
            
            if device is not None:
                inputs = inputs.cuda(device)

            features, bofs = model(inputs)

            loss = 0
            for feature, bof in zip(features, bofs):
                loss = loss + L(feature, bof)
           
            loss.backward()
            model_optimizer.step()

            train_loss += loss.item()
            counter += inputs.size(0)
        
        losses.append(train_loss / counter)
        print('epoch: %s' % str(epoch_offset + epoch_idx + 1))
        print('--loss: %.6f' % (train_loss / counter))

    return losses


def srm_init_epoch_trainer(target_type,
                           S_model,
                           T_model,
                           data_loader,
                           device,
                           lr=0.0001,
                           epochs=1,
                           epoch_offset=0,
                           weight_decay=1e-4,
                           verbose=0):

    optimizer = optim.Adam
    
    S_model.train()
    T_model.eval() 
    parameters = S_model.parameters()
    model_optimizer = optimizer(parameters, 
                                lr=lr, 
                                weight_decay=weight_decay)

    losses = []

    Llocal = CrossEntropyLoss()
    Lglobal = BCELoss() 

    for epoch_idx in range(epochs):
        if verbose:
            loader = tqdm(data_loader,
                    desc='epoch ' + str(epoch_offset+epoch_idx+1)+': ',
                          ncols=80,
                          ascii=True)
        else:
            loader = data_loader

        train_loss, counter =  0, 0
        
        if target_type == 'global-local':
            for inputs, _ in loader:
                model_optimizer.zero_grad()
                
                if device is not None:
                    inputs = inputs.cuda(device)

                s_outputs = S_model(inputs)
                s_gout1, s_gout2, s_gout3 = s_outputs[0]
                s_lout1, s_lout2, s_lout3 = s_outputs[1]

                with torch.no_grad():
                    t_outputs = T_model(inputs)
                t_gout1, t_gout2, t_gout3 = t_outputs[0]
                t_lout1, t_lout2, t_lout3 = t_outputs[1]
                
                loss = Lglobal(s_gout1, t_gout1.detach()) + \
                        Lglobal(s_gout2, t_gout2.detach()) + \
                        Lglobal(s_gout3, t_gout3.detach()) +\
                        Llocal(s_lout1, t_lout1.detach()) + \
                        Llocal(s_lout2, t_lout2.detach()) + \
                        Llocal(s_lout3, t_lout3.detach())
                
                loss.backward()
                model_optimizer.step()

                train_loss += loss.item()
                counter += inputs.size(0)
        else:
            L = Lglobal if target_type == 'global' else Llocal
            for inputs, _ in loader:
                model_optimizer.zero_grad()
                
                if device is not None:
                    inputs = inputs.cuda(device)

                s_outputs = S_model(inputs)
                s_out1, s_out2, s_out3 = s_outputs
                with torch.no_grad():
                    t_outputs = T_model(inputs)
                t_out1, t_out2, t_out3 = t_outputs
     
                loss = L(s_out1, t_out1.detach()) + \
                        L(s_out2, t_out2.detach()) + \
                        L(s_out3, t_out3.detach()) 
                
                loss.backward()
                model_optimizer.step()

                train_loss += loss.item()
                counter += inputs.size(0)

        losses.append(train_loss / counter)
        print('epoch: %s' % str(epoch_offset + epoch_idx + 1))
        print('--loss: %.6f' % (train_loss / counter))
        

    return losses 


def linear_classifier_epoch_trainer(model,
                                    data_loader,
                                    device,
                                    lr=0.0001,
                                    epochs=1,
                                    epoch_offset=0,
                                    weight_decay=1e-4,
                                    verbose=0):

    optimizer = optim.Adam
    
    model.train()
    parameters = model.output_layer.parameters() 
    model_optimizer = optimizer(parameters, lr=lr, weight_decay=weight_decay)

    losses = []
    accuracy = []

    L = CrossEntropyLoss()
    
    for epoch_idx in range(epochs):
        if verbose:
            loader = tqdm(data_loader,
                    desc='epoch ' + str(epoch_offset+epoch_idx+1)+': ',
                          ncols=80,
                          ascii=True)
        else:
            loader = data_loader

        nb_correct, train_loss, counter = 0, 0, 0
        
        for inputs, targets in loader:
            model_optimizer.zero_grad()
            
            if device is not None:
                inputs = inputs.cuda(device)
                targets = targets.cuda(device)

            outputs = model(inputs)

            targets = torch.squeeze(targets)
            loss = L(outputs, targets)
            
            acc = count_correct(outputs, targets)
           
            loss.backward()
            model_optimizer.step()

            train_loss += loss.item()
            counter += targets.size(0)
            nb_correct += acc.item() 

        losses.append(train_loss / counter)
        accuracy.append(nb_correct / counter)
        print('epoch: %s' % str(epoch_offset + epoch_idx + 1))
        print('--loss: %.6f' % (train_loss / counter))
        print('--acc: %.2f' % (nb_correct / counter)) 
    
    return losses, accuracy



def classifier_epoch_trainer(model,
                             data_loader,
                             device,
                             lr=0.0001,
                             epochs=1,
                             epoch_offset=0,
                             weight_decay=1e-4,
                             verbose=0):

    optimizer = optim.Adam
    
    model.train()
    parameters = model.parameters()
    model_optimizer = optimizer(parameters, 
                                lr=lr, 
                                weight_decay=weight_decay)

    losses = []
    accuracy = []

    L = CrossEntropyLoss()
    
    for epoch_idx in range(epochs):
        if verbose:
            loader = tqdm(data_loader,
                    desc='epoch ' + str(epoch_offset+epoch_idx+1)+': ',
                          ncols=80,
                          ascii=True)
        else:
            loader = data_loader

        nb_correct, train_loss, counter = 0, 0, 0
        
        for inputs, targets in loader:
            model_optimizer.zero_grad()
            
            if device is not None:
                inputs = inputs.cuda(device)
                targets = targets.cuda(device)

            outputs = model(inputs)

            targets = torch.squeeze(targets)
            loss = L(outputs, targets)
            
            acc = count_correct(outputs, targets)
           
            loss.backward()
            model_optimizer.step()

            train_loss += loss.item()
            counter += targets.size(0)
            nb_correct += acc.item() 

        losses.append(train_loss / counter)
        accuracy.append(nb_correct / counter)
        print('epoch: %s' % str(epoch_offset + epoch_idx + 1))
        print('--loss: %.6f' % (train_loss / counter))
        print('--acc: %.2f' % (nb_correct / counter)) 

    return losses, accuracy


def srm_init_evaluator(target_type, 
                       S_model,
                       T_model,
                       data_loader,
                       device,
                       verbose=False):

    S_model.eval()
    T_model.eval() 

    counter = 0
    Loss = 0
    Llocal = CrossEntropyLoss()
    Lglobal = BCELoss()

    with torch.no_grad():
        if verbose:
            loader = tqdm(data_loader,
                    desc='eval:',
                          ncols=80,
                          ascii=True)
        else:
            loader = data_loader
       
        if target_type == 'global-local':
            for inputs, _ in loader:
                if device is not None:
                    inputs = inputs.cuda(device)

                s_outputs = S_model(inputs)
                s_gout1, s_gout2, s_gout3 = s_outputs[0]
                s_lout1, s_lout2, s_lout3 = s_outputs[1]
                with torch.no_grad():
                    t_outputs = T_model(inputs)
                t_gout1, t_gout2, t_gout3 = t_outputs[0]
                t_lout1, t_lout2, t_lout3 = t_outputs[1]

                 
                counter += inputs.size(0)
                loss = Lglobal(s_gout1, t_gout1.detach()) + \
                        Lglobal(s_gout2, t_gout2.detach()) + \
                        Lglobal(s_gout3, t_gout3.detach()) +\
                        Llocal(s_lout1, t_lout1.detach()) + \
                        Llocal(s_lout2, t_lout2.detach()) + \
                        Llocal(s_lout3, t_lout3.detach())

                
                Loss += loss.item()
        else:
            L = Llocal if target_type == 'local' else Lglobal
            for inputs, _ in loader:
                if device is not None:
                    inputs = inputs.cuda(device)

                s_outputs = S_model(inputs)
                s_out1, s_out2, s_out3 = s_outputs
                with torch.no_grad():
                    t_outputs = T_model(inputs)
                t_out1, t_out2, t_out3 = t_outputs

                 
                counter += inputs.size(0)
                loss = L(s_out1, t_out1.detach()) + \
                        L(s_out2, t_out2.detach()) + \
                        L(s_out3, t_out3.detach()) 

                
                Loss += loss.item()

    Loss /= counter

    return Loss


def classifier_evaluator(model,
                         data_loader,
                         device,
                         verbose=False):

    model.eval()

    L = CrossEntropyLoss() 
    nb_correct = 0
    counter = 0
    loss = 0 
    with torch.no_grad():
        if verbose:
            loader = tqdm(data_loader,
                    desc='eval:',
                          ncols=80,
                          ascii=True)
        else:
            loader = data_loader
        
        for inputs, targets in loader:
            if device is not None:
                inputs = inputs.cuda(device)
                targets = targets.cuda(device)
            
            outputs = model(inputs)

            counter += inputs.size(0)
            
            acc = count_correct(outputs, targets)
            loss += L(outputs, targets).item()
            nb_correct += acc.item()

        accuracy = nb_correct / counter
        loss = loss / counter

    return loss, accuracy


def count_correct(output, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        k = 1
        batch_size = target.size(0)
        
        _, pred = output.topk(k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        
        return correct_k
