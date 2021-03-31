import os, sys, itertools
import exp_configurations as Conf
import train
import pickle

def create_configuration(names, hp_list):
    outputs_ = []

    for hp in hp_list:
        tmp = []
        for name in names:
            tmp.append(hp[name])
        outputs_ += list(itertools.product(*tmp))


    outputs = []
    for conf in outputs_:
        if conf not in outputs:
            outputs.append(conf)
    return outputs

def inspect():
    names = Conf.names

    configurations = create_configuration(names, [Conf.values_cifar100, Conf.values_tl])
    output_dir = Conf.OUTPUT_DIR
    missing = []
    for conf in configurations:
        filename = '_'.join([str(v) for v in conf]) + '.pickle'
        filename = os.path.join(output_dir, filename)
        if not os.path.exists(filename):
            missing.append(conf)

    return missing 

def main(argv):

    missings = inspect()
    for conf in missings:
        name = '_'.join([str(v) for v in conf])
        name = os.path.join(Conf.OUTPUT_DIR, name + '.pickle') 
        result = train.execute(conf)
        fid = open(name, 'wb')
        pickle.dump(result, fid)
        fid.close()


if __name__ == "__main__":
    main(sys.argv[1:])


