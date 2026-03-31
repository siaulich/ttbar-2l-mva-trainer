#!/bin/bash
executable = submitCondor.sh
universe = vanilla

mkdir -p logs
# Resource requirements
RequestGPUs    = 1
RequestMemory  = 60000
+RequestRuntime = 120000

# Output files
output = logs/job_$(Cluster)_$(ModelName).out
error = logs/job_$(Cluster)_$(ModelName).err
log = logs/job_$(Cluster)_$(ModelName).log

arguments = python3 ../TrainScript.py \
    --output_dir ../models/$(ModelName)/ \
    --load_config ../config/nominal_load_config.yaml \
    --train_config training/train_config.yaml \
    --model_config models/$(ModelName).yaml \
    --event_numbers even 


# Queue from file - reads each line and assigns to variables
queue ModelName from $(ModelNamesFile)