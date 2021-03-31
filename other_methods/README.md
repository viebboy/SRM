**This repository is based on [CRD repo](https://github.com/HobbitLong/RepDistiller)**
---------------------------------------------------------------------------------------

To run experiments, given "root" refer to the root directory of this repository

- data should be downloaded and put under "root/data"
- first download the pretrained teachers from [here](https://drive.google.com/file/d/1UYvvRMEGu1fMY0emzPPGENSTr75keOlw/view?usp=sharing) and put under directory "root/other_methods/save/models"
- to run all models (AT, PKT, RKD, CRD) and configurations, simply run "python train_all.py"
- to test if code is working, modify the variable "mode" in "exp_configurations.py" to value "test". This will run the code for only 3 epochs. Otherwise, set "mode" to "deploy" to run the full experiments 
