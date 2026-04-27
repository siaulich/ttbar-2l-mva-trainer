# ttbar-2l-mva-trainer
A standalone framework for training and evaluating machine learning models for reconstruction of dileptonic ttbar events. The framework includes data-preprocessing, model training using TensorFlow, evaluation and integration with the Condor job scheduler for distributed training and evaluation.

The framework is designed to provide models, that can be directly used for evaluation or ntuple production via [TopCPToolKit](https://topcptoolkit.docs.cern.ch/latest/settings/reconstruction/#dilepassigner). 

## Setup

The code can be run in a virtual environment. To set up the virtual environment, you can run the following commands:

```bash
python -m venv venv \
source venv/bin/activate \
pip install -r requirements.txt \
```
Note, that the default installation of TensorFlow might not support GPU acceleration.

## Preprocessing 
The preprocessing step is responsible for converting `.root` files into `.npz` files, which can be used for training and evaluation. The preprocessing step is configured using the `examples/preprocessing.yaml` file, which specifies the name of the tree and branches to be read from the `.root` files, as well as the name of the output `.npz` file. The preprocessing step can be run using the `scripts/PreprocessScript.py` script.

```
python scripts/PreProcessing.py \
--name nominal \
--config examples/preprocessing.yaml \
--input_dir /path/to/root/files \
--output_dir dilep_data \
```


## Training
The training step is responsible for training machine learning models using the preprocessed `.npz` files. This step requires 3 configuration files:

- `examples/load_nominal_config.yaml`: This file specifies the configuration for loading the preprocessed data.\
`data_path: SetMe`: The path to the preprocessed `.npz` files has to be set here.
- `examples/compact_assigner.yaml`: This file specifies the configuration for the machine learning model to be trained, including the architecture and hyperparameters.
- `examples/training_config.yaml`: This file specifies the configuration for the training process, including the number of epochs, batch size, and evaluation metrics.

The training step can be run using the `scripts/TrainScript.py` script.

```
python scripts/TrainScript.py \
--load_config examples/load_nominal_config.yaml \
--model_config examples/compact_assigner.yaml \
--train_config examples/train_config.yaml \
--output_dir models \
--event_numbers even \
--num_events 1000000 \
```

## Evaluation
The evaluation step is responsible for evaluating the performance of the trained machine learning models using the preprocessed `.npz` files. This step requires 2 configuration files:
- `examples/load_nominal_config.yaml`: This file specifies the configuration for loading the preprocessed data.\
`data_path: SetMe`: The path to the preprocessed `.npz` files has to be set here.
- `examples/evaluation_config.yaml`: This file specifies the configuration for the evaluation process, including the evaluation metrics to be used.

The evaluation step can be run using the `scripts/EvaluateScript.py` script.

```
python scripts/EvaluateScript.py \
--load_config examples/load_nominal_config.yaml \
--evaluation_config examples/evaluation_config.yaml \
--output_dir plots/evaluation \
--event_numbers odd \
--num_events 1000000 \
```


## Export to ONNX
The trained machine learning models can be exported to the ONNX format for deployment and inference. When running the training script, the trained model will be automatically exported to the ONNX format and saved in the specified output directory. To use the exported ONNX model for inference, it is recommended to train two separate models, one for even and one for odd event numbers. This allows to avoid any potential bias in the training data.