#!/bin/bash
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --account=terrafirma
#SBATCH -o %x.%j.o
#SBATCH -e %x.%j.e
#SBATCH --time=24:00:00
#SBATCH --mem=64GB

# Timeseries - update existing files, or calculate from scratch
python -c "from nemo_python.projects.evaluation import *; update_timeseries_evaluation_NEMO_AIS('./')"

# Time-averaged fields - remove if they exist, then calculate from scratch
for f in bottom_TS_avg.nc zonal_TS_avg.nc; do
    if [ -f $f ]; then
	rm $f
    fi
done
python -c "from nemo_python.projects.evaluation import *; precompute_avg(option='bottom_TS', out_file='bottom_TS_avg.nc')"
python -c "from nemo_python.projects.evaluation import *; precompute_avg(option='zonal_TS', out_file='zonal_TS_avg.nc')"
