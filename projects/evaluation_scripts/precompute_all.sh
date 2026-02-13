#!/bin/bash
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --account=terrafirma
#SBATCH -o %x.%j.o
#SBATCH -e %x.%j.e
#SBATCH --time=24:00:00
#SBATCH --mem=64GB

# Timeseries and Hovmollers - update existing files, or calculate from scratch
python -c "from nemo_python.projects.evaluation import *; update_timeseries_evaluation_NEMO_AIS('./')"
python -c "from nemo_python.projects.evaluation import *; update_hovmollers_evaluation_NEMO_AIS('./')"

# Time-averaged fields - remove if they exist, then calculate from scratch
OPTIONS=( bottom_TS zonal_TS seaice ismr vel )
for OPTION in "${OPTIONS[@]}"; do
    filename = ${OPTION}_avg.nc
    if [ -f $filename; then
	rm $filename
    fi
    python -c "from nemo_python.projects.evaluation import *; precompute_avg(option='"${OPTION}"')"
done
	 
