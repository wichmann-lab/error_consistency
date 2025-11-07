#!/bin/bash

# run this as . setup.sh to make GPUs available

if [ -d "/.singularity.d/env" ]; then
    for f in /.singularity.d/env/*; do source "$f"; done
fi

nvidia-smi