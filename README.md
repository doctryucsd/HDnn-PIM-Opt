# HDnn-RRAM-Opt
The code for optimization is in `sim/`. `sim/` includes the optimization algorithms and the simulator for HDnn-PIM. `plots/` includes scripts to plot results.

## Setup
This repo includes various external submodules. To clone this repo to run please do 
````
git clone --recurse-submodules https://github.com/doctryucsd/HDnn-RRAM-Opt.git
````

The code needs to run in a docker container of `timeloopaccelergy/timeloop-accelergy-pytorch:latest-amd64`. To setup the docker container, please follow the instructions in https://github.com/mit-emze/cimloop. Please note that the docker container needs to support cuda.

## Run Optimization
To run the optimization, run
````
python -m sim
````
The results will in be `outputs/`.

To change the settings, please modify `conf/config.yaml`.