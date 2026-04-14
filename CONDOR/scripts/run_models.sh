#!/bin/bash
executable = submitCondor.sh
universe = vanilla

# Resource requirements
RequestGPUs    = 1
RequestMemory  = 16000
+RequestRuntime = 36000

# Output files
output = logs/job_$(Cluster)_$(ModelName)_$(ConfigName).out
error = logs/job_$(Cluster)_$(ModelName)_$(ConfigName).err
log = logs/job_$(Cluster)_$(ModelName)_$(ConfigName).log

arguments = python3 ../TrainScript.py \
    --output_dir ../models/$(ModelName)_$(ConfigName)/ \
    --load_config ../config/load_$(ConfigName).yaml \
    --train_config training/train_config.yaml \
    --model_config models/$(ModelName).yaml \
    --event_numbers even 


# Queue from file - reads each line and assigns to variables
queue ModelName, ConfigName from $(ModelNamesFile)