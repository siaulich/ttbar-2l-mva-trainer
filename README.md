# ttbar-2l-mva-trainer
A standalone framework for training and evaluating machine learning models for reconstruction of dileptonic ttbar events. The framework includes data-preprocessing, data loading and model training using TensorFlow, and integration with the Condor job scheduler for distributed training and evaluation.

The models are designed to perform both reconstruction of the neutrino momenta as well as assignment of jets to the corresponding b-quarks from the top quark decays.

To inject the trained machine learning models into the TopCPToolKit, the models can be exported to ONNX format. Currently, only models feed-forward neural network architectures are supported for export to ONNX format, but support for additional architectures can be implemented.

While the framework is designed for training and evaluating machine learning models for ATLAS, it can be easily adapted for use in other contexts by modifying the data preprocessing and loading steps to fit the specific requirements of the new context. The modular design of the framework allows for easy integration of new models and evaluation metrics.


## Setup

The code can be run in a virtual environment. To set up the virtual environment, you can run the following commands:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
Note, that the default installation of TensorFlow might not support GPU acceleration.

## Preprocessing 
The preprocessing step is responsible for converting `.root` files into `.npz` files, which can be used for training and evaluation. The preprocessing step is configured using the `examples/preprocessing.yaml` file, which specifies the name of the tree and branches to be read from the `.root` files, as well as the name of the output `.npz` file. The preprocessing step can be run using the `scripts/PreprocessScript.py` script.

```
python scripts/PreprocessScript.py 
--name nominal 
--config examples/preprocessing.yaml 
--input_dir /path/to/root/files 
--output_dir dilep_data/nominal.npz
```


## Training
The training step is responsible for training machine learning models using the preprocessed `.npz` files. This step requires 3 configuration files:

- `examples/load_nominal_config.yaml`: This file specifies the configuration for loading the preprocessed data.\
`data_path: SetMe`: The path to the preprocessed `.npz` files has to be set here.
- `examples/compact_assigner.yaml`: This file specifies the configuration for the machine learning model to be trained, including the architecture and hyperparameters.
- `examples/training_config.yaml`: This file specifies the configuration for the training process, including the number of epochs, batch size, and evaluation metrics.

The training step can be run using the `scripts/TrainScript.py` script.

```
python scripts/TrainScript.py 
--load_config examples/load_nominal_config.yaml 
--model_config examples/compact_assigner.yaml 
--training_config examples/training_config.yaml 
--output_dir models/
--event_numbers even
--num_events 1000000
```

## Evaluation
The evaluation step is responsible for evaluating the performance of the trained machine learning models using the preprocessed `.npz` files. This step requires 2 configuration files:
- `examples/load_nominal_config.yaml`: This file specifies the configuration for loading the preprocessed data.\
`data_path: SetMe`: The path to the preprocessed `.npz` files has to be set here.
- `examples/evaluation_config.yaml`: This file specifies the configuration for the evaluation process, including the evaluation metrics to be used.

The evaluation step can be run using the `scripts/EvaluateScript.py` script.

```
python scripts/EvaluateScript.py 
--load_config examples/load_nominal_config.yaml 
--evaluation_config examples/evaluation_config.yaml 
--output_dir /path/to/output/evaluation
--event_numbers odd
--num_events 1000000
```
