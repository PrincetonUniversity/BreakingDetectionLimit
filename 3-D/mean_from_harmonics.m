function [M, G] = mean_from_harmonics(a_vec, j_0, s0, L)
% Evaluate the mean of the volume from volume expansion coefficients.
% See reconstruct_from_clean_autocorrs_script.m for example usage.
% 
% Inputs:
%   * a_vec: vecotrized volume expansion coefficients
%   * j_0: cell array of zeroth-order spherical bessel functions (see
%   generate_spherical_bessel_basis.m)
%   * s0: number of zeroth-order radial frequencies
%   * L: length of volume (or projection)
% 
% Outputs:
%   * M: mean of the volume
%   * G: gradient of M with respect to a_vec

G = [j_0(:); zeros(length(a_vec)-s0,1)]./(L^3*sqrt(4*pi));
M = G.'*a_vec;