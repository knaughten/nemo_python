### ND, 17.07.26

(thanks to Alethea Mountford, Birgit Rogalla and Kaitlin Naughten for very helpful advice)

This code takes UKESM (eORCA01) ocean + sea ice output and produces boundary conditions for CANOBI (eORCA025).

To run the code, set the parameters (suite ID, path, startyear, endyear) in config.py, then run make_bdy_cond.sh.

The code should then automatically:

1) Check whether the input data is there
2) produce boundary files for the variables: Conservative Temperature, Absolute Salinity, UVEL, VVEL, SSH, sea ice area and height, snow height

The code produces one file per variable per time interval (e.g., month).

It takes about 70 seconds per year to process on a JASMIN standard partition node.

The velocity regridding is agnostic to volume conservation. There are, however, NEMO4.2.2 flags that (if the manual is to be believed) correct the boundary conditions so that volume is conserved.

CANOBI starts running with the produced files, but I have not done any long-term stability tests.

There are many things to be improved and sped up; I'll try to do that over time.
