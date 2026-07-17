#!/bin/bash
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --account=terrafirma
#SBATCH -o %x.%j.o
#SBATCH -e %x.%j.e
#SBATCH --time=01:00:00
#SBATCH --mem=64GB

python make_bdy_cond.py
