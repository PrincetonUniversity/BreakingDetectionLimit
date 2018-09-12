## Autocorrelations From Micrographs for Cryo-EM
### Computation from micrographs and volume reconstruction

#### Dependencies:
1. [ASPIRE](http://spr.math.princeton.edu/)
2. [Manopt](http://manopt.org/)
3. [Spherical Harmonic Transform toolbox by Archontis Politis](https://www.mathworks.com/matlabcentral/fileexchange/43856-real-complex-spherical-harmonic-transform--gaunt-coefficients-and-rotations)
4. [EasySpin](https://www.easyspin.org/)
5. [Ab initio Kam's method toolbox](https://github.com/eitangl/kam_cryo)

#### Main functions:

1. **reconstruct_from_clean_autocorrs_script.m**: Script to load a volume, generate its clean first three autocorrelations, and inverts them to recover the volume. 
2. **moments_from_micrographs_script.m**: Script to load micrographs and compute the average first three autocorrelations of the dataset.
3. **pure_noise_moments_script.m**: Script to compute autocorrelations from pure-noise micrographs.
4. **pure_noise_detection_script.m**: After running scripts 2. & 3., run this script to recover the predicted number of projections per micrograph, and hence to detect pure noise micrographs.
5. **precomp_B_factors_script.m**: Script to precompute quantities needed for bispectrum evaluation.

##### Misc. Code
We include code in trispectrum_code which was not used for the paper, but might be useful for future extensions. 
In particular, they include methods to compute the bispectrum and trispectrum by using projections, which can be scaled to high resolution.

In case of issues or questions, please email Eitan Levin (eitanl@math.princeton.edu)
