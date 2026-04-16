#!/bin/bash
# submit_kfold.sh

MODEL_NAMES_FILE=$1

{
# Print the static part of the submit file
cat <<'EOF'
executable = submitCondor.sh
universe = vanilla
RequestGPUs    = 1
RequestMemory  = 16000
+RequestRuntime = 36000
output = logs/job_$(Cluster)_$(ModelName)_$(ConfigName)_fold$(KFoldIndex).out
error = logs/job_$(Cluster)_$(ModelName)_$(ConfigName)_fold$(KFoldIndex).err
log = logs/job_$(Cluster)_$(ModelName)_$(ConfigName)_fold$(KFoldIndex).log
arguments = python3 ../TrainKFoldScript.py \
--output_dir ../models/$(ModelName)_$(ConfigName)_fold$(KFoldIndex)/ \
--load_config ../config/load_$(ConfigName).yaml \
--train_config training/train_config.yaml \
--model_config models/$(ModelName).yaml \
--event_numbers even \
--k_fold 5 \
--k_fold_index $(KFoldIndex)

queue ModelName, ConfigName, KFoldIndex from (
EOF

# Expand each model/config line into 5 fold entries
while IFS=' ,' read -r modelname configname; do
  for i in 0 1 2 3 4; do
    echo "  $modelname $configname $i"
  done
done < "$MODEL_NAMES_FILE"

echo ")"
} | condor_submit