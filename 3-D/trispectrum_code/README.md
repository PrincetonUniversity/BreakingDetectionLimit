## Autocorrelations From Micrographs for Cryo-EM
### Misc. Code

#### Main functions:

1. 'reconstruct_volume_from_trispectrum.m': Script to load volume, compute its first $four$ autocorrelations, and invert to recover 
the volume. It uses the precomputation scheme used for the paper for the bispectrum, but a projection-based approach for the trispectrum. 
The parallelization scheme for the trispectrum is naive and is done on the CPU - ideally we would use multiple GPUs, but currently managing the memory is a problem.
2. 'reconstruct_volume_from_trispectrum_slice_GPUs.m': Script to load volume, compute its first $three$ autocorrelations and a 4-tensor slice of the (5-tensor) trispectrum, 
and invert to recover the volume. Here both bispectrum and trispectrum slice are computed using the projection-based approach which scales to high resolution, and multiple GPUs are
utilized for this. However, this trispectrum slice does not seem to solve the ill-conditioning problem.

In case of issues or questions, please email Eitan Levin (eitanl@math.princeton.edu)
