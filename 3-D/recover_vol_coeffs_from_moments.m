function [a_lms, gamma] = recover_vol_coeffs_from_moments(a3_micro, a2_micro, a1_micro, maxL, L, B, W, B_lists, r_cut, a_init)
% Function to solve a least-squares problem to recover the volume expansion
% coefficients and the average fraction of pixels in a micrograph occupied
% by signal.
% 
% Inputs:
%   * a3_micro, a2_micro, a1_micro: first three autocorrelations computed
%   from the micrographs
%   * maxL: cutoff for spherical harmonics expansion
%   * L: length of volume or projection
%   * B, W, B_lists: precomputed quantities for bispectrum evaluation (see
%   precomp_B_factors_script.m)
%   * r_cut: assumed bandlimit (Nyquist is 1/2)
%   * a_init: initial guess, omit or leave empty to use a random guess
% 
% Outputs:
%   * a_lms: cell array of recovered volume expansion coefficients
%   * gamma: recovered average fraction of pixels on a micrograph occupied
%   by signal.
% 
% Eitan Levin, August 2018

% Get vectorization indices for the coefficients:
x_lists.s = gen_s_list(maxL, r_cut, floor(L/2));
[x_lists.L, x_lists.m, ~, x_lists.p, x_lists.n] = gen_vec_coeff_lists(maxL, x_lists.s);

% Get bases and quadrature for 1st and 2nd moment computations:
q0 = sqrt(sum(B_lists.blk_id == 1));
[r,w] = lgwt(20*q0, 0, 1);
M12_quants.w = w.*r;

M12_quants.j_l = generate_spherical_bessel_basis(maxL, x_lists.s, 1/2, (1/2)*r);
M12_quants.j_0 = cell2mat(generate_spherical_bessel_basis(0, x_lists.s, 1/2, 0));

[R_0n, alpha_Nn_2D] = PSWF_radial_2D(0, q0-1, pi*(L-1), r); % generate 2D radial prolates
M12_quants.R_0n = bsxfun(@times, R_0n, 2./alpha_Nn_2D(:).');

% Initialization:
if ~exist('a_init','var') || isempty(a_init)
    a_init = randn(2*length(x_lists.p), 1);
    a_init(end+1) = 0.5;
end

% Bounds for occupancy factor:
lb = zeros(2*length(x_lists.p)+1, 1, 'double');
lb(1:end-1) = -inf;
lb(end) = 0;

ub = zeros(2*length(x_lists.p)+1, 1, 'double');
ub(1:end-1) = inf;
ub(end) = 1;

% Weights for least-squares (lambda(1) -> 3rd moment, lambda(2) -> 2nd
% moment, lambda(3) -> 1st moment)

% lambda(1) = 1;
% lambda(2) = sqrt(numel(m3_micro)/numel(m2_micro));
% lambda(3) = sqrt(numel(m3_micro)/numel(m1_micro));

% lambda(1) = 1/norm(m3_micro);
% lambda(2) = 1/norm(m2_micro);
% lambda(3) = 1/norm(m1_micro);

lambda = [1,1,1];

lambda

costgrad = @(x) costgrad_lsqnonlin(x, x_lists, L, B, W, B_lists, M12_quants, a3_micro, a2_micro, a1_micro, lambda, maxL);

iters_per_save = 500;
save_name = ['bispect_precomp_maxL_' num2str(maxL)];
saveFunc = @(x, optimValues, state) generic_saveFunc(x, optimValues, state, save_name, iters_per_save);

options = optimoptions('lsqnonlin','Jacobian','on','DerivativeCheck','off','Display','iter','TolX',1e-20,'TolFun',1e-20,'MaxIter',1e4, 'MaxFunEvals', 1e5, 'OutputFcn', saveFunc);

tic, x = lsqnonlin(costgrad, a_init, lb, ub, options); time_TR = toc

gamma = x(end);
a_lms = x(1:(end-1)/2) + 1i*x((end-1)/2+1:end-1);
a_lms = vec_to_cell_vol_coeffs(a_lms, x_lists.L(x_lists.m>=0));
