#!/bin/bash

# Check if command argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <command>"
    echo "Example: $0 'python3 scripts/train_regression_transformer.py'"
    exit 1
fi

COMMAND="$*"

mkdir -p logs

# Create temporary HTCondor submit file
SUBMIT_FILE=$(mktemp /tmp/condor_submit.XXXXXX)

cat > "$SUBMIT_FILE" << EOF
#!/bin/bash
# HTCondor submission file for hyperparameter grid search
executable = submitCondor.sh
universe = vanilla

# Resource requirements
RequestGPUs    = 1
RequestMemory  = 16000
+RequestRuntime = 36000

# Output files
output = logs/job_\$(Cluster).out
error = logs/job_\$(Cluster).err
log = logs/job_\$(Cluster).log

arguments = $COMMAND

# Queue from file - reads each line and assigns to variables
queue 1
EOF

condor_submit "$SUBMIT_FILE"

# Clean up
rm "$SUBMIT_FILE"