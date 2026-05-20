#!/bin/bash
executable = evaluate_all.sh
universe = vanilla

# Resource requirements
RequestGPUs    = 1
RequestMemory  = 40000
+RequestRuntime = 36000

# Output files
output = logs/job_$(Cluster).out
error = logs/job_$(Cluster).err
log = logs/job_$(Cluster).log



# Queue from file - reads each line and assigns to variables
queue 1
