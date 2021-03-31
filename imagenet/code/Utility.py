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


def kd_epoch_trainer(S_model,
                    T_model,
                    optim_state,
                    data_loader,
                    device,
                    lr=0.0001,
                    epochs=1,
                    epoch_offset=0,
                    weight_decay=1e-4,
                    alpha=0.5,
                    temperature=1.0,
                    verbose=0,
                    multigpu=True):

    optimizer = optim.SGD

    S_model.train()
    T_model.eval()
    parameters = S_model.parameters()
    model_optimizer = optimizer(parameters, 
                                momentum=0.9,
                                nesterov=True,
                                lr=lr, 
                                weight_decay=weight_decay)
    
    if optim_state is not None:
        model_optimizer.load_state_dict(optim_state)
        print('using previous optimizer state')
    else:
        print('using new optimizer state')


    losses = []
    accuracy1 = []
    accuracy5 = []

    
    for epoch_idx in range(epochs):
        if verbose:
            loader = tqdm(data_loader,
                    desc='epoch ' + str(epoch_offset+epoch_idx+1)+': ',
                          ncols=80,
                          ascii=True)
        else:
            loader = data_loader

        nb_correct1, nb_correct5, train_loss, counter = 0, 0, 0, 0
        
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
            
            acc1, acc5 = count_correct(outputs, targets, topk=(1, 5))
           
            loss.backward()
            model_optimizer.step()

            train_loss += loss.item()
            counter += targets.size(0)
            nb_correct1 += acc1.item() 
            nb_correct5 += acc5.item()

        losses.append(train_loss / counter)
        accuracy1.append(nb_correct1 / counter)
        accuracy5.append(nb_correct5 / counter)
 
        print('epoch: %s' % str(epoch_offset + epoch_idx + 1))
        print('--loss: %.6f' % (train_loss / counter))
        print('--acc top1: %.2f' % (nb_correct1 / counter)) 
        print('--acc top5: %.2f' % (nb_correct5 / counter)) 
    
    model_optimizer.zero_grad()
    
    return losses, accuracy1, accuracy5, model_optimizer.state_dict()


def knowledge_distil(logfile,
                    S_model,
                    T_model,
                    compute,
                    train_loader,
                    val_loader,
                    LR,
                    Epochs,
                    weight_decay,
                    alpha,
                    temperature,
                    verbose,
                    multigpu):

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
        losses = data['losses']
        accuracy1 = data['accuracy1']
        accuracy5 = data['accuracy5']
        val_measure = data['val_measure']
        optim_state = data['optim_state']

    else:
        losses = []
        accuracy1 = []
        accuracy5 = []
        epoch_offset = 0
        val_measure = 0
        optim_state = None

    if compute == 'gpu':
        device = torch.device('cuda')
        S_model.cuda(device)
        T_model.cuda(device)
        torch.backends.cudnn.benchmark = True
    else:
        device = None


    for idx, (lr, epochs) in enumerate(zip(LR, Epochs)):
        losses_, accuracy_1, accuracy_5, optim_state = \
                kd_epoch_trainer(S_model,
                                  T_model, 
                                  optim_state,
                                  train_loader,
                                  device,
                                  lr,
                                  epochs,
                                  epoch_offset,
                                  weight_decay,
                                  alpha,
                                  temperature,
                                  verbose,
                                  multigpu)

        losses += losses_
        accuracy1 += accuracy_1
        accuracy5 += accuracy_5

        epoch_offset += epochs

        # evaluate on train set
        val_acc, _ = classifier_evaluator(S_model, 
                                          val_loader, 
                                          device, 
                                          verbose)
        print('val acc top1: %.2f' % val_acc)

        if val_acc > val_measure:
            val_measure = val_acc
            best_model_state = S_model.cpu().state_dict() 

        fid = open(model_file, 'wb')
        pickle.dump(best_model_state, fid)
        fid.close()

        S_model.cuda(device)
        
        # reset optimizer
        if idx + 1 < len(LR) and lr != LR[idx+1]:
            optim_state = None
            print('resetting optimizer')
        # log data
        weights = S_model.cpu().state_dict()

        data = {'model_state': weights,
                'LR': LR[idx+1:],
                'Epochs': Epochs[idx+1:],
                'epoch_offset': epoch_offset,
                'losses': losses,
                'accuracy1': accuracy1,
                'accuracy5': accuracy5,
                'val_measure': val_measure,
                'optim_state': optim_state}

        fid = open(logfile, 'wb')
        pickle.dump(data, fid)
        fid.close()
        S_model.cuda(device)
    
    
    fid = open(model_file, 'rb')
    best_model_state = pickle.load(fid)
    fid.close()

    S_model.cpu()
    S_model.load_state_dict(best_model_state)
    S_model.cuda(device)

    train_acc_top1, train_acc_top5 = classifier_evaluator(S_model, 
                                                          train_loader, 
                                                          device, 
                                                          verbose)
    val_acc_top1, val_acc_top5 = classifier_evaluator(S_model, 
                                                      val_loader, 
                                                      device, 
                                                      verbose)

    print('train acc top1: %.2f' % train_acc_top1)
    print('train acc top5: %.2f' % train_acc_top5)
    print('val acc top1: %.2f' % val_acc_top1)
    print('val acc top5: %.2f' % val_acc_top5)

    S_model.cpu()

    outputs = {'model_weights': S_model.state_dict(),
               'train_acc_top1': train_acc_top1,
               'val_acc_top1': val_acc_top1,
               'train_acc_top5': train_acc_top5,
               'val_acc_top5': val_acc_top5,
               'train_loss': losses,
               'train_accuracy1': accuracy1,
               'train_accuracy5': accuracy5}

    return outputs

def train_codeword(logdir,
                    model,
                    device,
                    data_loader,
                    LR,
                    Epochs,
                    weight_decay,
                    verbose,
                    multigpu):

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
        optim_state = data['optim_state']

    else:
        losses = []
        epoch_offset = 0
        val_measure = 1e10
        optim_state = None

    model.cuda(device)


    for idx, (lr, epochs) in enumerate(zip(LR, Epochs)):
        losses_, optim_state  = codeword_epoch_trainer(model,
                                                       optim_state,
                                                       data_loader,
                                                       device,
                                                       lr,
                                                       epochs,
                                                       epoch_offset,
                                                       weight_decay,
                                                       verbose,
                                                       multigpu)

        losses += losses_

        epoch_offset += epochs

        # if train loss improve, save the weights
        if losses[-1] < val_measure: 
            val_measure = losses[-1]
            model.cpu()
            best_model_state = model.state_dict()
            fid = open(model_file, 'wb')
            pickle.dump({'model_weights':best_model_state}, fid)
            fid.close()
            model.cuda(device)

        # reset optimizer state if change learning rate
        if idx + 1 < len(LR) and lr != LR[idx+1]:
            print('resetting optimizer state')
            optim_state = None
        
        # log data
        weights = model.cpu().state_dict()
        data = {'model_state': weights,
                'LR': LR[idx+1:],
                'Epochs': Epochs[idx+1:],
                'epoch_offset': epoch_offset,
                'losses': losses,
                'val_measure': val_measure,
                'optim_state': optim_state}

        fid = open(logfile, 'wb')
        pickle.dump(data, fid)
        fid.close()
        model.cuda(device)
    
    model.cpu() 
    fid = open(model_file, 'rb')
    best_model_state = pickle.load(fid)['model_weights']
    fid.close()

    model.load_state_dict(best_model_state)
    model.cpu()

   
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
             LR,
             Epochs,
             weight_decay,
             verbose,
             multigpu):
    
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
        optim_state = data['optim_state']

    else:
        losses = []
        epoch_offset = 0
        val_measure = 1e10
        optim_state = None

    S_model.cuda(device)
    T_model.cuda(device)

    torch.backends.cudnn.benchmark = True


    for idx, (lr, epochs) in enumerate(zip(LR, Epochs)):
        losses_, optim_state = srm_init_epoch_trainer(target_type,
                                                      S_model,
                                                      T_model,
                                                      optim_state,
                                                      train_loader,
                                                      device,
                                                      lr,
                                                      epochs,
                                                      epoch_offset,
                                                      weight_decay,
                                                      verbose,
                                                      multigpu)

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

        # reset optimizer state
        if idx+1 < len(LR) and lr != LR[idx+1]:
            optim_state = None
            print('resetting optimizer state')

        weights = S_model.cpu().state_dict()

        data = {'model_state': weights,
                'LR': LR[idx+1:],
                'Epochs': Epochs[idx+1:],
                'epoch_offset': epoch_offset,
                'losses': losses,
                'val_measure': val_measure,
                'optim_state': optim_state}

        fid = open(logfile, 'wb')
        pickle.dump(data, fid)
        fid.close()
        S_model.cuda(device)
    
    fid = open(model_file, 'rb')
    best_model_state = pickle.load(fid)['model_weights']
    fid.close()

    S_model.cpu()
    S_model.load_state_dict(best_model_state)


    outputs = {'model_weights': S_model.state_dict(),
               'train_loss': losses}

    return outputs


def codeword_epoch_trainer(model,
                           optim_state,
                           data_loader,
                           device,
                           lr=0.0001,
                           epochs=1,
                           epoch_offset=0,
                           weight_decay=1e-4,
                           verbose=0,
                           multigpu=False):

    optimizer = optim.Adam

    
    model.train()
    # if codeword trainable, only train the codewords
    # otherwise, train other layers but with fixed codewords
    
    if multigpu:
        parameters = model.module.sparse_layers.parameters()
        model_optimizer = optimizer(parameters, 
                                lr=lr, 
                                weight_decay=weight_decay)
    else:
        parameters = model.sparse_layers.parameters()
        model_optimizer = optimizer(parameters, 
                                lr=lr, 
                                weight_decay=weight_decay)
        
    if optim_state is not None:
        print('loading old optimizer state')
        model_optimizer.load_state_dict(optim_state)
    else:
        print('--train with new optimizer state')

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
       
        iteration = 0
        for inputs, _ in loader:
            model_optimizer.zero_grad()
            
            inputs = inputs.to(device)

            features, reps = model(inputs)

            loss = 0
            for feature, rep in zip(features, reps):
                loss = loss + L(feature, rep)
          

            loss.backward()
            model_optimizer.step()

            train_loss += loss.item()
            counter += inputs.size(0)
       
            iteration += 1

        losses.append(train_loss / counter)
        print('epoch: %s' % str(epoch_offset + epoch_idx + 1))
        print('--loss: %.6f' % (train_loss / counter))


    model_optimizer.zero_grad()
    
    return losses, model_optimizer.state_dict()


def srm_init_epoch_trainer(target_type,
                            S_model,
                            T_model,
                            optim_state,
                            data_loader,
                            device,
                            lr=0.0001,
                            epochs=1,
                            epoch_offset=0,
                            weight_decay=1e-4,
                            verbose=0,
                            multigpu=False):

    optimizer = optim.SGD

    
    S_model.train()
    T_model.eval() 
    parameters = S_model.parameters()
    model_optimizer = optimizer(parameters, 
                                lr=lr, 
                                weight_decay=weight_decay,
                                momentum=0.9,
                                nesterov=True)

    if optim_state is not None:
        model_optimizer.load_state_dict(optim_state)
        print('using previous optimizer state')
    else:
        print('using new optimizer')

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
                s_gouts, s_louts = s_outputs

                with torch.no_grad():
                    t_outputs = T_model(inputs)

                t_gouts, t_louts = t_outputs
    
                loss = 0
                for s_gout, t_gout in zip(s_gouts, t_gouts):
                    loss = loss + Lglobal(s_gout, t_gout.detach())
                for s_lout, t_lout in zip(s_louts, t_louts):
                    loss = loss + Llocal(s_lout, t_lout.detach())
                
                
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
                for s_out, t_out in zip(s_outputs, t_outputs):
                    loss = loss + L(s_out, t_out)

                loss.backward()
                model_optimizer.step()

                train_loss += loss.item()
                counter += inputs.size(0)

        losses.append(train_loss / counter)
        print('epoch: %s' % str(epoch_offset + epoch_idx + 1))
        print('--loss: %.6f' % (train_loss / counter))
 
        

    model_optimizer.zero_grad()

    return losses, model_optimizer.state_dict() 



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
                with torch.no_grad():
                    t_outputs = T_model(inputs)

                s_gouts, s_louts = s_outputs
                t_gouts, t_louts = t_outputs

                loss = 0
                
                for s_gout, t_gout in zip(s_gouts, t_gouts):
                    loss = loss + Lglobal(s_gout, t_gout.detach())

                for s_lout, t_lout in zip(s_louts, t_louts):
                    loss = loss + Llocal(s_lout, t_lout.detach())

                counter += inputs.size(0)
                Loss += loss.item()
        else:
            L = Llocal if target_type == 'local' else Lglobal
            for inputs, _ in loader:
                if device is not None:
                    inputs = inputs.cuda(device)

                s_outputs = S_model(inputs)
                with torch.no_grad():
                    t_outputs = T_model(inputs)
                   
                loss = 0
                for s_out, t_out in zip(s_outputs, t_outputs):
                    loss = loss + L(s_out, t_out)

                counter += inputs.size(0)
                Loss += loss.item()

    Loss /= counter

    return Loss



def classifier_evaluator(model,
                          data_loader,
                          device,
                          verbose=False):

    model.eval()

    nb_correct1 = 1
    nb_correct5 = 1

    counter = 0
    
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
            
            acc1, acc5 = count_correct(outputs, targets, topk=(1, 5))

            nb_correct1 += acc1.item()
            nb_correct5 += acc5.item()


        accuracy1 = nb_correct1 / counter
        accuracy5 = nb_correct5 / counter


    return accuracy1, accuracy5 




def count_correct(output, target, topk=(1, 5)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        correct_k = []
        for k in topk:
            correct_k.append(correct[:k].view(-1).float().sum(0, keepdim=True))
        
        return correct_k
