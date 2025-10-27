#!/bin/bash

# Generate the initial conditions if they are not present.
if [ ! -e glassCube_64.hdf5 ]
then
    echo "Fetching initial glass file for the Sod shock example..."
    ./getGlass.sh
fi
if [ ! -e sedov.hdf5 ]
then
    echo "Generating initial conditions for the Sod shock example..."
    python3 makeIC.py
fi

# Run SWIFT
../../../swift_cuda --pin --hydro --threads=8 sedovGPU.yml 2>&1 | tee outputGPU.log

# Plot the solution
python3 plotSolution.py 5
