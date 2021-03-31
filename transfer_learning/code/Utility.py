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
                    device,
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

    S_model.cuda(device)
    T_model.cuda(device)
    torch.backends.cudnn.benchmark = True


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
   
        S_model.to(device)
   
    S_model.cpu()
    T_model.cpu()
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
            
            inputs = inputs.cuda(device)
            targets = targets.cuda(device)

            s_outs = S_model(inputs)
            with torch.no_grad():
                t_outs = T_model(inputs)

            loss = 0 
            for s_out, t_out in zip(s_outs, t_outs):
                loss = loss + L(s_out, t_out)
           
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
    train_acc = []
    test_loss = []
    test_acc = []
    
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

        train_loss_, train_acc_ = classifier_evaluator(S_model,
                                                       train_loader,
                                                       device,
                                                       verbose)
        test_loss_, test_acc_ = classifier_evaluator(S_model,
                                                       test_loader,
                                                       device,
                                                       verbose)

        S_model.train()

        train_loss.append(train_loss_)
        train_acc.append(train_acc_)
        test_loss.append(test_loss_)
        test_acc.append(test_acc_)

        print('epoch: %s' % str(epoch_offset + epoch_idx + 1))
        print('--train loss: %.6f' % train_loss_)
        print('--train acc: %.6f' % train_acc_)
        print('--test loss: %.6f' % test_loss_)
        print('--test acc: %.6f' % test_acc_)


    return train_loss, train_acc, test_loss, test_acc 




def knowledge_distil(logfile,
                    S_model,
                    T_model,
                    device,
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
        val_measure = data['val_measure']
        train_loss = data['train_loss']
        train_accuracy = data['train_accuracy']
        test_loss = data['test_loss']
        test_accuracy = data['test_accuracy']

    else:
        epoch_offset = 0
        val_measure = 0
        train_loss = []
        test_loss = []
        train_accuracy = []
        test_accuracy = []

    S_model.cuda(device)
    T_model.cuda(device)
    torch.backends.cudnn.benchmark = True

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

        train_loss_, train_acc_, test_loss_, test_acc_ = epoch_outputs

        train_loss += train_loss_
        train_accuracy += train_acc_
        test_loss += test_loss_
        test_accuracy += test_acc_

        epoch_offset += epochs

        if val_loader is not None:
            # evaluate on train set
            _, val_acc = classifier_evaluator(S_model, 
                                              val_loader, 
                                              device, 
                                              verbose)
            print('val acc: %.4f' % val_acc)

            if val_acc > val_measure:
                val_measure = val_acc
                best_model_state = S_model.cpu().state_dict() 
                fid = open(model_file, 'wb')
                pickle.dump(best_model_state, fid)
                fid.close()
        else:
            best_model_state = S_model.cpu().state_dict()
            fid = open(model_file, 'wb')
            pickle.dump(best_model_state, fid)
            fid.close()

        
        # log data
        weights = S_model.cpu().state_dict()
        data = {'model_state': weights,
                'LR': LR[idx+1:],
                'Epochs': Epochs[idx+1:],
                'epoch_offset': epoch_offset,
                'val_measure': val_measure,
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                'test_loss': test_loss,
                'test_accuracy': test_accuracy}

        fid = open(logfile, 'wb')
        pickle.dump(data, fid)
        fid.close()
        if device is not None: 
            S_model.to(device)

    fid = open(model_file, 'rb')
    best_model_state = pickle.load(fid)
    fid.close()

    S_model.cpu()
    S_model.load_state_dict(best_model_state)
    S_model.cuda(device)

    _, train_acc = classifier_evaluator(S_model, 
                                        train_loader, 
                                        device, 
                                        verbose)
    _, test_acc = classifier_evaluator(S_model, 
                                       test_loader, 
                                       device, 
                                       verbose)

    if val_loader is not None:
        _, val_acc = classifier_evaluator(S_model, 
                                          val_loader, 
                                          device, 
                                          verbose)
    else:
        val_acc = 0

    print('train acc: %.4f' % train_acc)
    print('val acc: %.4f' % val_acc)
    print('test acc: %.4f' % test_acc)

    S_model.cpu()
    outputs = {'model_weights': S_model.state_dict(),
               'train_acc': train_acc,
               'val_acc': val_acc,
               'test_acc': test_acc,
               'train_loss': train_loss,
               'train_accuracy': train_accuracy,
               'test_loss': test_loss,
               'test_accuracy': test_accuracy}

    return outputs


def train_codeword(logfile,
                    model_file,
                    model,
                    multigpu,
                    device,
                    data_loader,
                    LR,
                    Epochs,
                    verbose):

    model = model.float()
    model.to(device) 
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


    for idx, (lr, epochs) in enumerate(zip(LR, Epochs)):
        losses_  = codeword_epoch_trainer(model,
                                        multigpu,
                                        data_loader,
                                        device,
                                        lr,
                                        epochs,
                                        epoch_offset,
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
             device,
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

    
    S_model.to(device)
    T_model.to(device)
    torch.backends.cudnn.benchmark = True

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
                S_model.cuda(device)
        else:
            best_model_state = S_model.cpu().state_dict()
            fid = open(model_file, 'wb')
            pickle.dump({'model_weights':best_model_state}, fid)
            fid.close()
            S_model.cuda(device)

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
        S_model.cuda(device)
    
    fid = open(model_file, 'rb')
    best_model_state = pickle.load(fid)['model_weights']
    fid.close()

    S_model.cpu()
    S_model.load_state_dict(best_model_state)
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

    S_model.cpu()
    T_model.cpu()
    outputs = {'model_weights': S_model.state_dict(),
               'train_loss': train_loss,
               'val_loss': val_loss,
               'test_loss': test_loss,
               'train_losses': losses}

    return outputs




def train_classifier(logfile,
                    model,
                    device,
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
        train_loss = data['train_loss']
        train_accuracy = data['train_accuracy']
        test_loss = data['test_loss']
        test_accuracy = data['test_accuracy']
        val_measure = data['val_measure']

    else:
        epoch_offset = 0
        val_measure = 0
        train_loss = []
        train_accuracy = []
        test_loss = []
        test_accuracy = []

    model.to(device)
    torch.backends.cudnn.benchmark = True

    for idx, (lr, epochs) in enumerate(zip(LR, Epochs)):
        epoch_outputs = classifier_epoch_trainer(model,
                                                train_loader,
                                                test_loader,
                                                device,
                                                lr,
                                                epochs,
                                                epoch_offset,
                                                weight_decay,
                                                verbose)

        train_loss_, train_accuracy_, test_loss_, test_accuracy_ = epoch_outputs
        train_loss += train_loss_
        train_accuracy += train_accuracy_
        test_loss += test_loss_
        test_accuracy += test_accuracy_

        epoch_offset += epochs

        if val_loader is not None:
            # evaluate on train set
            _, val_acc = classifier_evaluator(model, 
                                              val_loader, 
                                              device, 
                                              verbose)
            print('val acc: %.4f' % val_acc)

            if val_acc > val_measure:
                val_measure = val_acc
                best_model_state = model.cpu().state_dict()
                fid = open(model_file, 'wb')
                pickle.dump(best_model_state, fid)
                fid.close()
                model.to(device)

        else:
            best_model_state = model.cpu().state_dict()
            fid = open(model_file, 'wb')
            pickle.dump(best_model_state, fid)
            fid.close()

            model.cuda(device)

        # log data
        weights = model.cpu().state_dict()
        data = {'model_state': weights,
                'LR': LR[idx+1:],
                'Epochs': Epochs[idx+1:],
                'epoch_offset': epoch_offset,
                'train_accuracy': train_accuracy,
                'train_loss': train_loss,
                'test_accuracy': test_accuracy,
                'test_loss': test_loss,
                'val_measure': val_measure}

        fid = open(logfile, 'wb')
        pickle.dump(data, fid)
        fid.close()
        model.to(device)

    fid = open(model_file, 'rb')
    best_model_state = pickle.load(fid)
    fid.close()

    model.cpu()
    model.load_state_dict(best_model_state)
    model.cuda(device)

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

    print('train acc: %.4f' % train_acc)
    print('val acc: %.4f' % val_acc)
    print('test acc: %.4f' % test_acc)

    outputs = {'model_weights': model.state_dict(),
               'train_accuracy': train_accuracy,
               'train_loss': train_loss,
               'test_accuracy': test_accuracy,
               'test_loss': test_loss,
               'train_acc': train_acc,
               'test_acc': test_acc,
               'val_acc': val_acc}

    return outputs

def codeword_epoch_trainer(model,
                          multigpu,
                          data_loader,
                          device,
                          lr=0.0001,
                          epochs=1,
                          epoch_offset=0,
                          verbose=0):

    optimizer = optim.Adam
    
    model.train()
    # if codeword trainable, only train the codewords
    # otherwise, train other layers but with fixed codewords
    if multigpu:
        parameters = model.module.sparse_layers.parameters()
    else:
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
                
                inputs = inputs.cuda(device)

                s_gouts, s_louts = S_model(inputs)

                with torch.no_grad():
                    t_gouts, t_louts = T_model(inputs)
                
                loss = 0
                
                for s_gout, t_gout in zip(s_gouts, t_gouts):
                    loss = loss + Lglobal(s_gout, t_gout)

                for s_lout, t_lout in zip(s_louts, t_louts):
                    loss = loss + Llocal(s_lout, t_lout)
                
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
                
                with torch.no_grad():
                    t_outputs = T_model(inputs)
     
                loss = 0 
                for sout, tout in zip(s_outputs, t_outputs):
                    loss = loss + L(sout, tout)
                
                loss.backward()
                model_optimizer.step()

                train_loss += loss.item()
                counter += inputs.size(0)

        losses.append(train_loss / counter)
        print('epoch: %s' % str(epoch_offset + epoch_idx + 1))
        print('--loss: %.6f' % (train_loss / counter))
        

    return losses 



def classifier_epoch_trainer(model,
                             train_loader,
                             test_loader,
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

    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    L = CrossEntropyLoss()
    
    for epoch_idx in range(epochs):
        if verbose:
            loader = tqdm(train_loader,
                    desc='epoch ' + str(epoch_offset+epoch_idx+1)+': ',
                          ncols=80,
                          ascii=True)
        else:
            loader = train_loader

        counter = 0
        
        for inputs, targets in loader:
            model_optimizer.zero_grad()
            
            if device is not None:
                inputs = inputs.cuda(device)
                targets = targets.cuda(device)

            outputs = model(inputs)

            targets = torch.squeeze(targets)
            loss = L(outputs, targets)
           
            loss.backward()
            model_optimizer.step()

            counter += targets.size(0)

        train_loss_, train_acc_ = classifier_evaluator(model,
                                                     train_loader,
                                                     device,
                                                     verbose)
        test_loss_, test_acc_ = classifier_evaluator(model,
                                                   test_loader,
                                                   device,
                                                   verbose)

        model.train()
        train_loss.append(train_loss_)
        train_acc.append(train_acc_)
        test_loss.append(test_loss_)
        test_acc.append(test_acc_)

        print('epoch: %s' % str(epoch_offset + epoch_idx + 1))
        print('--test loss: %.6f' % (test_loss_))
        print('--test acc: %.4f' % (test_acc_)) 

    return train_loss, train_acc, test_loss, test_acc



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
                inputs = inputs.cuda(device)

                s_gouts, s_louts = S_model(inputs)
                with torch.no_grad():
                    t_gouts, t_louts = T_model(inputs)

                counter += inputs.size(0)
                loss = 0 
                for sgout, tgout in zip(s_gouts, t_gouts):
                    loss = loss + Lglobal(sgout, tgout)
                for slout, tlout in zip(s_louts, t_louts):
                    loss = loss + Llocal(slout, tlout)

                                
                Loss += loss.item()
        else:
            L = Llocal if target_type == 'local' else Lglobal
            for inputs, _ in loader:
                if device is not None:
                    inputs = inputs.cuda(device)

                s_outputs = S_model(inputs)
                with torch.no_grad():
                    t_outputs = T_model(inputs)

                counter += inputs.size(0)
                loss = 0 
                for s_out, t_out in zip(s_outputs, t_outputs):
                    loss = loss + L(s_out, t_out)
                
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

            nb_correct += acc.item()
            loss += L(outputs, targets).item()

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
