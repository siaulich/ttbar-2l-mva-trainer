#!/bin/bash
# HTCondor submission file for hyperparameter grid search
executable = submitCondor.sh
universe = vanilla

# Resource requirements
RequestGPUs    = 1
RequestMemory  = 60000
+RequestRuntime = 60000

# Output files
output = logs/job_eval_$(Cluster).out
error = logs/job_eval_$(Cluster).err
log = logs/job_eval_$(Cluster).log

arguments = python3 ../EvaluateScript.py \
    --output_dir ../evaluation/$(ModelName)/ \
    --load_config ../config/load_test_config.yaml \
    --evaluation_config evaluation/evaluation_config.yaml \
    --model_config models/$(ModelName).yaml \
    --event_numbers $(EventNumbers)


# Queue from file - reads each line and assigns to variables
queue 