# SkyLens

Dependencies

1. Dask: To enable parallel computation using graphs.
2. Boltzmann code to compute Power spectra: We support Camb, Class and CCL. There are wrapper functions in power_spectra.py. 
In the __init__ you can choose the default function to call and compute power spectra, can also be passed as argument through power_spectra_kwargs dictionary. In case you get import error, please comment out the import lines in power_spectra.py.
3. sympy: To compute wigner_3j matrices. 
4. sparse, zarr: This is used to effciently store and read some large wigner_3j matrices, which are computed only once.
5. Healpy: For window related calcuations, we use healpy maps (window inputs are healpy maps).
6. Astropy, numpy, scipy.
