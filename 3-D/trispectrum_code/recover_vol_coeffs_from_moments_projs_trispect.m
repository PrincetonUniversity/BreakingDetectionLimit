function [a_lms, gamma] = recover_vol_coeffs_from_moments_projs_trispect...
(m4_micro, m3_micro, m2_micro, m1_micro, M4_quants, B, W, B_lists, M12_quants,...
x_lists, maxL, L, Rots, k_max, a_init)
% Invert the first four autocorrelations to recover the volume expansion
% coefficients and the average fraction of pixels in the micrograph
% occupied by signal.
% 
% Inputs:
%   * m4_micro, m3_micro, m2_micro, m1_micro: first four autocorrelations
%   computed from the micrographs
%   * M4_quants: quantities for trispectrum computation (see
%   precomp_for_autocorrs_from_projs.m)
%   * B, W, B_lists: quantities for bispectrum computation (see
%   reconstruct_volume_from_trispectrum.m)
%   * M12_quants: quantities for mean and power spectrum computation
%   * x_lists: lists of indices for the vectorization of volume expansion
%   coefficients (see reconstruct_volume_from_trispectrum.m)
%   * maxL: cutoff for spherical harmonics expansion
%   * L: length of volume or projection
%   * Rots: stack of 3x3 rotations representing viewing directions with
%   which to compute the trispectrum
%   * k_max: maximum angular frequency of the trispectrum
%   * a_init: initial guess for coefficients. Omit or leave empty to use
%   random initialization.
% 
% Outputs:
%   * a_lms: cell array of volume expansion coefficients
%   * gamma: average fraction of micrograph pixels occupied by signal
% 
% Eitan Levin, August 2018

% Weights for least-squares (lambda(1) -> trispectrum, lambda(2) ->
% bispectrum, lambda(3) -> power spectrum, lambda(4) -> mean)

lambda(1) = 1;
lambda(2) = sqrt(numel(m4_micro)/numel(m3_micro));
lambda(3) = sqrt(numel(m4_micro)/numel(m2_micro));
lambda(4) = sqrt(numel(m4_micro)/numel(m1_micro));

lambda

% Bounds for occupancy factor:
lb = zeros(2*length(x_lists.p)+1, 1, 'double');
lb(1:end-1) = -inf;
lb(end) = 0;

ub = zeros(2*length(x_lists.p)+1, 1, 'double');
ub(1:end-1) = inf;
ub(end) = 1;

iters_per_save = 10; % save current iterate every X iterations
% First use only bispectrum:
if ~exist('a_init','var') || isempty(a_init)
    a_init = randn(2*length(x_lists.p), 1);
    a_init(end+1) = 0.5;

    % First use only bispectrum:
    costgrad = @(x) costgrad_lsqnonlin(x, x_lists, L, B, W, B_lists, M12_quants, m3_micro, m2_micro, m1_micro, lambda(2:end), maxL);

    saveFunc = @(x, optimValues, state) generic_saveFunc(x, optimValues, state, 'bispect_for_trispect', iters_per_save);
    options = optimoptions('lsqnonlin', 'Jacobian', 'on', 'DerivativeCheck', 'off', 'Display', 'iter', 'TolX',1e-4, 'TolFun',1e-4, 'MaxIter', 1e4, 'OutputFcn', saveFunc);

    tic, a_init = lsqnonlin(costgrad, a_init, lb, ub, options); time_bispect = toc
end

% Add slice of trispectrum:
% costgrad = @(x) costgrad_lsqnonlin_projs_trispect_projPar(x, x_lists, L, M12_quants, B, W, B_lists, M4_quants, m4_micro, m3_micro, m2_micro, m1_micro, lambda, gamma, k_max, maxL);

costgrad = @(x) costgrad_lsqnonlin_projs_trispect_projPar_withGamma(x, x_lists, L, M12_quants, B, W, B_lists, M4_quants, m4_micro, m3_micro, m2_micro, m1_micro, lambda, k_max, maxL);

save_name = ['trispect_projPar_projs_' num2str(size(Rots,3)) '_maxL_' num2str(maxL)];
saveFunc = @(x, optimValues, state) generic_saveFunc(x, optimValues, state, save_name, iters_per_save);
options = optimoptions('lsqnonlin', 'Jacobian', 'on', 'DerivativeCheck', 'off', 'Display', 'iter', 'TolX',1e-4, 'TolFun',1e-4, 'MaxIter', 1e4, 'OutputFcn', saveFunc);

tic, x = lsqnonlin(costgrad, a_init, lb, ub, options); time_TR = toc

gamma = x(end);
a_lms = x(1:(end-1)/2) + 1i*x((end+1)/2:end-1);
a_lms = vec_to_cell_vol_coeffs(a_lms, x_lists.L(x_lists.m>=0));
