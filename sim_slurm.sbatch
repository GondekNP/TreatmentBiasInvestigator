#!/bin/bash

#SBATCH --job-name=mpi_gondek_LSC_proj
#SBATCH --ntasks=30
#SBATCH --partition=broadwl
#SBATCH --account=macs30123

# Load Python and MPI modules
module load python/anaconda-2019.03
module load intelmpi/2018.2.199+intel-18.0

python3 ./fit_model.py
python3 ./gen_data.py

# Run the python program with mpirun, using & to run jobs at the same time
mpirun -n 30 python3 ./combine.py > ./sim_verbose.out &

# Wait until all simultaneous mpiruns are done
wait
