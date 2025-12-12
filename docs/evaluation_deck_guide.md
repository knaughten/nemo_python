# Introduction

Here are instructions for how to generate a bunch of figures to evaluate the simulated Southern Ocean and Antarctic continental shelf against observations. This currently includes:

1. a 24-panelled timeseries plot showing ice shelf basal melting, bottom temperature, and bottom salinity in 8 regions of the continental shelf. Grey lines are monthly model output, black lines are the 2-year running mean, and the blue dashed lines and shaded regions are observational estimates including uncertainty (Adusumilli 2020 for basal melting, Zhou 2025 for T and S):

(timeseries_shelf.png)

2. a 3-panelled timeseries plot showing transport through Drake Passage, the Weddell and Ross Gyres. Observational estimates are from Donohue 2016, Klatt 2005, and Dotto 2018:

(timeseries_transport.png)

3. a 6-panelled map showing bottom temperature salinity in the model, observations (Zhou 2025), and the difference:

(bottom_TS.png)

4. a 6-panelled slice plot showing zonally averaged temperature and salinity in the model, observations (WOA 2023), and the difference:

(to add)

More figures are planned so check back often for updates.

This code has been tested on [Birgit Rogalla's circum-Antarctic NEMO configuration](https://github.com/brogalla/eANT025-AntArc). An option will soon be added for UKESM1, and eventually also UKESM2.

In order to use Kaitlin's precomputed gridded observations files, it must be run on JASMIN, and you must have access to the `terrafirma` group workspace. If you want to run on another system, ask Kaitlin for a copy of these files.

# 1. Download and prepare the code

Clone this repository into a directory of your choice:

      cd <some_dir>
      git clone git@github.com:knaughten/nemo_python.git

Add the following line to your `~/.bashrc` file:

     export PYTHONPATH=$PYTHONPATH:<some_dir>

Now, get the top-level evaluation scripts in the same directory as your NEMO output files. If you want to make changes to the scripts (eg comment out some figures, or change the budget for the queue), make a copy:

     cp <some_dir>/nemo_python/projects/evaluation_scripts/* <nemo_output_dir>

If you don't want to change anything, make a link instead (this should then automatically update to new versions if you do `git pull` on the `nemo_python` repository):

   ln -s <some_dir>/nemo_python/projects/evaluation_scripts/* <nemo_output_dir>

# 2. Precompute files for plotting

In `<nemo_output_dir>`, submit a job script to the SLURM queue:

   sbatch precompute_all.sh

This will precompute three things:

1. Timeseries from the model output (producing files `timeseries_T.nc` and `timeseries_U.nc`)
2. Bottom temperature and salinity time-averaged over the last 20 years of simulation (producing `bottom_TS_avg.nc`)
3. Zonally averaged temperature and salinity time-averaged over the last 20 years of simulation (producing `zonal_TS_avg.nc`)

If you run this job script again after NEMO has run for longer, it will update the timeseries files from #1 as needed with any new NEMO files. However, it will delete the time-averaged files from #2 and #3, and remake them from scratch.

Depending on how many years you're trying to process, this can be slow. It's probably best to set it off to finish overnight, but if you're in a hurry you could split up the three steps into different jobs to run simultaneously.

# 3. Make the plots

In `<nemo_output_dir>`, run:

   python plot_all.py

This will create a few files `evaluation_*.png` corresponding to the evaluation figures shown above.

Enjoy!