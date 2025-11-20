#!/bin/bash

 # Generate the initial conditions if they are not present.
if [ ! -e EAGLE_ICs_12.hdf5 ]
then
    echo "Fetching initial conditions for the EAGLE 12Mpc example..."
    ./getIC.sh
fi

../../../swift_cuda --hydro --self-gravity --threads=16 eagle_12_GPU.yml 2>&1 | tee output.log

