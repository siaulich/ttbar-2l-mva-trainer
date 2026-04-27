#!/bin/bash

echo "Working dir: $PWD"
source ../venv/bin/activate

echo "Command: $@"
eval "$@"