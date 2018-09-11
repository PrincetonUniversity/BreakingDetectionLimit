function [F,J] = costgrad_lsqnonlin_projs_trispect(x, ...
    x_lists, L, M12_quants, B, W, B_lists, M4_quants, m4_micro, m3_micro,...
    m2_micro, m1_micro, lambda, k_max, L_cutoff)
% Evaluate cost and gradient of the cost, including first four
% autocorrelations from the micrographs. 
% 
% Inputs:
%   * x: current iterate. If a_lms is the vectorized coefficients of the 
%   volume and gamma is the fraction of pixels on the micrograph occupied 
%   by signal, x satisfes x = [real(a_lms); imag(a_lms); gamma].
%   * x_lists: list of indices for the vectorized volume coefficients (see
%   reconstruct_from_clean_autocorres_script.m for example)
%   * L: length of the volume (or projection)
%   * M12_quants: precomputed quantities for power spectrum and mean
%   computations
%   * B: precomputed factors for bispectrum evaluation (see
%   precomp_bispectrum_coeffs_matMult.m)
%   * W: precomputed weights involving Wigner 3j symbols (see
%   precomp_wigner_weights.m)
%   * B_lists: lists of indices for vectorization in B and the bispectrum
%   (see precomp_bispectrum_coeffs_matMult.m)
%   * M4_quants: precomputed quantities for trispectrum evaluation (see
%   reconstruct_volume_from_trispectrum.m)
%   * m4_micro, m3_micro, m2_micro, m1_micro: trispectrum, bispectrum, 
%   power spectrum and mean computed from the micrograph (see 
%   moments_from_micrograph_steerable.m)
%   * lambda: weights for the three terms in the cost, where lambda(1)
%   being the weight of the trispectrum difference, lambda(3) for bispectrum, 
%   lambda(2) for power spectrum, and lambda(1) for the mean.
%   * k_max: cutoff for angular frequency of the trispectrum
%   * L_cutoff: cutoff for spherical harmonics expansion
% 
% Outputs:
%   * F: vector such that the cost is given by norm(F)^2
%   * J: Jacobian of F
% 
% Eitan Levin, August 2018

% parse optimization variable:
gamma = x(end);
a_lms = x(1:(end-1)/2) + 1i*x((end+1)/2:end-1);
a_lms = vec_to_cell_vol_coeffs(a_lms, x_lists.L(x_lists.p));
a_vec = vec_cell(a_lms);

% Compute 1st and 2nd moments:
[m2_harms, G_m2] = power_spectrum_from_harmonics(a_vec, M12_quants.j_l, M12_quants.R_0n, M12_quants.w, L, L_cutoff);
dm2 = gamma*m2_harms - m2_micro;

[m1_harms, G_m1] = mean_from_harmonics(a_vec, M12_quants.j_0, x_lists.s(1), L);
dm1 = gamma*m1_harms - m1_micro;

% Compute gradient of trispectrum slice and bispectrum:
[G_m4, m4_harms] = trispectrum_grad_from_harmonics_projs_projPar(a_lms, M4_quants.psi_coeffs_lNs, M4_quants.psi_lNs, M4_quants.q_list, M4_quants.D_mats, L, k_max);

dm4 = gamma*m4_harms - m4_micro;

[G1, G2, G3] = bispectrum_grad_from_harmonics_par_noKloop...
        (a_lms, L, B, W, B_lists.L1, B_lists.L2, B_lists.L3, x_lists.L, x_lists.m, x_lists.s, B_lists.blk_id);
    
m3_harms = real(G1.'*a_vec);
dm3 = gamma*m3_harms - m3_micro;
G_m3 = G1 + G2 + conj(G3);

F = [lambda(1)*real(dm4); lambda(2)*real(dm3); lambda(3)*real(dm2); lambda(4)*real(dm1)];
    
sign_factor = (-1).^(x_lists.L(x_lists.n)+x_lists.m(x_lists.n));

J4 = bsxfun(@times, G_m4(x_lists.n,:), sign_factor) + conj(G_m4(x_lists.p, :));
J3 = bsxfun(@times, G_m3(x_lists.n,:), sign_factor) + conj(G_m3(x_lists.p, :));
J2 = 2*(G_m2(x_lists.p,:) + bsxfun(@times, conj(G_m2(x_lists.n, :)), sign_factor));

J = [lambda(1)*J4, lambda(2)*J3, lambda(3)*J2]; % add m2 derivative
J(x_lists.m(x_lists.p) == 0,:) = (1/2)*J(x_lists.m(x_lists.p) == 0, :); % treat m = 0 case
    
J = [J, lambda(4)*G_m1(x_lists.p)]; % add m1 derivative

J = gamma*[real(J); imag(J)].'; % separate real and imaginary parts
   
J = [J, real([lambda(1)*m4_harms; lambda(2)*m3_harms; lambda(3)*m2_harms; lambda(4)*m1_harms])]; % add gamma derivative
    

