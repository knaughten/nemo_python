# ND, 17.07.26

This code takes UKESM (eORCA01) ocean + sea ice output and produces boundary conditions for CANOBI (eORCA025).

To run the code, set the parameters (suite ID, path, startyear, endyear) in config.py, then run make_bdy_cond.sh.

The code should then automatically:

1) check whether the input data is there
2) produce boundary files for the variables: Conservative Temperature, Absolute Salinity, UVEL, VVEL, SSH, sea ice area and height, snow height

The code produces one file per variable per time interval (e.g., month).

It takes about 70s per year processed on a JASMIN standard partition node.

The velocity regridding is agnostic to volume conservation. There are, however, NEMO4.2.2 flags that (if the manual is to be believed) correct the boundary conditions so that volume is conserved ()

CANOBI starts running with the produced files, but I have not done longer stablity tests.

There are many things to be improved, sped up, I'll try to do that over time.