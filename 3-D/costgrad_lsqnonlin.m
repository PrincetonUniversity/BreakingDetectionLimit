function [F, J] = costgrad_lsqnonlin(x, x_lists, L, B, W, B_lists,...
    M2_quants, m3_micro, m2_micro, m1_micro, lambda, maxL)
% Compute vector F such that the objective is given by norm(F)^2, and the
% jacobian J of F.
% 
% Inputs:
%   * x: current iterate. If a_lms is the vectorized coefficients of the 
%   volume and gamma is the fraction of pixels on the micrograph occupied 
%   by signal, x satisfes x = [real(a_lms); imag(a_lms); gamma].
%   * x_lists: list of indices for the vectorized volume coefficients (see
%   reconstruct_from_clean_autocorres_script.m for example)
%   * L: length of the volume (or projection)
%   * B: precomputed factors for bispectrum evaluation (see
%   precomp_bispectrum_coeffs_matMult.m)
%   * W: precomputed weights involving Wigner 3j symbols (see
%   precomp_wigner_weights.m)
%   * B_lists: lists of indices for vectorization in B and the bispectrum
%   (see precomp_bispectrum_coeffs_matMult.m)
%   * M2_quants: precomputed quantities for power spectrum and mean
%   computations
%   * m3_micro, m2_micro, m1_micro: bispectrum, power spectrum and mean
%   computed from the micrograph (see moments_from_micrograph_steerable.m)
%   * lambda: weights for the three terms in the cost, where lambda(1)
%   being the weight of the bispectrum difference, lambda(2) for power
%   spectrum, and lambda(1) for the mean.
%   * maxL: cutoff for spherical harmonics expansion
% 
% Outputs:
%   * F: vector such that the cost is given by norm(F)^2
%   * J: Jacobian of F
% 
% Eitan Levin, August 2018

% Calculate objective f
gamma = x(end);
a_lms = x(1:(end-1)/2) + 1i*x((end-1)/2+1:end-1);
a_lms = vec_to_cell_vol_coeffs(a_lms, x_lists.L(x_lists.p));
a_vec = vec_cell(a_lms);

[m2_harms, G_m2] = power_spectrum_from_harmonics(a_vec, M2_quants.j_l, M2_quants.R_0n, M2_quants.w, L, maxL);
dm2 = gamma*m2_harms - m2_micro;

[m1_harms, G_m1] = mean_from_harmonics(a_vec, M2_quants.j_0, x_lists.s(1), L);
dm1 = gamma*m1_harms - m1_micro;

if nargout == 1 % cost only
    
    m3_harms = bispectrum_from_harmonics(a_lms, B, W, B_lists.L1, B_lists.L2, B_lists.L3, x_lists.s, B_lists.blk_id);
    dm3 = gamma*m3_harms - m3_micro;
    
    F = [lambda(1)*real(dm3); lambda(2)*real(dm2); lambda(3)*real(dm1)];
    
else % Jacobian required
    [G1, G2, G3] = bispectrum_grad_from_harmonics...
        (a_lms, B, W, B_lists.L1, B_lists.L2, B_lists.L3, x_lists.L, x_lists.m, x_lists.s, B_lists.blk_id);
    
    m3_harms = G1.'*a_vec;
    
    dm3 = gamma*m3_harms - m3_micro;
    F = [lambda(1)*real(dm3); lambda(2)*real(dm2); lambda(3)*real(dm1)];
    
    % Impose real-valued volume, so the coefficients satisfy
    % a_{l, -m, s} = (-1)^(l+m) * conj(a_{l, m, s})
    sign_factor = (-1).^(x_lists.L(x_lists.n)+x_lists.m(x_lists.n));
    
    J = G1 + G2;
    J = bsxfun(@times, J(x_lists.n,:), sign_factor) + G3(x_lists.p, :);
    
    J = [lambda(1)*J, 2*lambda(2)*G_m2(x_lists.p,:)]; % add m2 derivative
    
    J(x_lists.m(x_lists.p) > 0,:) = 2*J(x_lists.m(x_lists.p) > 0, :); % treat m = 0 case
    
    J = [J, lambda(3)*G_m1(x_lists.p)]; % add m1 derivative
    J = gamma*[real(J); imag(J)].'; % separate real and imaginary parts
    
    J = [J, real([lambda(1)*m3_harms; lambda(2)*m2_harms; lambda(3)*m1_harms])]; % add gamma derivative
    
end
