# HTCondor Support for ttbar-2l-mva-trainer
This directory contains scripts and configuration files for running the ttbar-2l-mva-trainer framework on the HTCondor job scheduler.

## Setup
To run the ttbar-2l-mva-trainer framework on HTCondor, you need to set the correct path to the virtual environment in the script `submitCondor.sh`. 

You can also customize the directory path for the output of the training and evaluation results in the corresponding configuration submission scripts in the `scripts` directory.

## Scripts
-  `run_models.sh`: This script is used to submit training jobs for multiple models and configurations to HTCondor. It reads the model names and configuration names from a specified input file and submits a separate job for each combination of model and configuration.
- `run_k_fold_model.sh`: This script is used to submit training jobs for k-fold cross-validation to HTCondor. It reads the model names and configuration names from a specified input file and submits a separate job for each combination of model, configuration, and fold index.
- `run_full_training.sh`: This script is used to submit a full training job for a single model and configuration to HTCondor. It reads the model name and configuration name from a specified input file and submits two separate jobs for even and odd event numbers to avoid any potential bias in the training data. It produces an output directory with the trained models, that can be directly used for evaluation or ntuple production via [TopCPToolKit](https://topcptoolkit.docs.cern.ch/latest/settings/reconstruction/#dilepassigner).