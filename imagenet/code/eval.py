import os, sys, getopt
import exp_configurations as Conf
import Models
import torch
import pickle
import Datasets
import Utility

def main(argv):
    # get model definition
    model = Models.resnet18(pretrained=False) 
    
    # load weights
    fid = open(os.path.join(Conf.DATA_DIR, 'srm_resnet18_pretrained.pickle'), 'rb')
    model_weights = pickle.load(fid)
    fid.close()
    model.load_state_dict(model_weights)

    # move to GPU
    device = torch.device('cuda')    
    model.to(device)

    # get validation data
    _, val_loader = Datasets.get_data('imagenet', 128)

    top1, top5 = Utility.classifier_evaluator(model,
                                              val_loader,
                                              device,
                                              verbose=True)

    print('top1 accuracy: %.5f' % top1)
    print('top5 accuracy: %.5f' % top5)



if __name__ == "__main__":
    main(sys.argv[1:])

