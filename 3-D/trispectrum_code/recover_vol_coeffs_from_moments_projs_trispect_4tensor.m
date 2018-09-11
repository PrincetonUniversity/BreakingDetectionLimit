function [a_lms, gamma] = recover_vol_coeffs_from_moments_projs_trispect_4tensor...
                    (m4_micro, m3_micro, m2_micro, m1_micro, maxL, L, r_cut, a_init, Rots, q_cutoff)

% Get vectorization indices for the coefficients:
x_lists.s = gen_s_list(maxL, r_cut, 1, floor(L/2));
[x_lists.L, x_lists.m, ~, x_lists.p, x_lists.n] = gen_vec_coeff_lists(maxL, x_lists.s);

% Precompute quantities for 3rd moment:

%Rots = sample_S2(num_pts);

[M34_quants.psi_curr, M34_quants.curr_freqs, M34_quants.psi_lNs, M34_quants.q_list, M34_quants.D_mats, M34_quants.psi_freqs, M34_quants.a_sizes, M34_quants.psi_curr_k0] = precomp_for_bispect_from_projs(maxL, x_lists.s, L, Rots);

% Get bases and quadrature for 1st and 2nd moment computations:
q0 = M34_quants.q_list(1);
[r,w] = lgwt(40*q0, 0, 1);
M12_quants.w = w.*r;

M12_quants.j_l = generate_spherical_bessel_basis(maxL, x_lists.s, 1/2, (1/2)*r);
M12_quants.j_0 = cell2mat(generate_spherical_bessel_basis(0, x_lists.s, 1/2, 0));

[R_0n, alpha_Nn_2D] = PSWF_radial_2D(0, q0-1, pi*(L-1), r); % generate 2D radial prolates
M12_quants.R_0n = bsxfun(@times, R_0n, 2./alpha_Nn_2D(:).');

% Weights for least-squares (lambda(1) -> 3rd moment, lambda(2) -> 2nd
% moment, lambda(3) -> 1st moment)

lambda(1) = 1;
lambda(2) = sqrt(numel(m4_micro)/numel(m3_micro));
lambda(3) = sqrt(numel(m4_micro)/numel(m2_micro));
lambda(4) = sqrt(numel(m4_micro)/numel(m1_micro));

lambda

iters_per_save = 10; % save current iterate every X iterations
% First use only bispectrum:
if ~exist('a_init','var') || isempty(a_init)
    a_init = randn(2*length(x_lists.p), 1);
    a_init(end+1) = 0.5;


    % Bounds for occupancy factor:
    lb = zeros(2*length(x_lists.p)+1, 1, 'double');
    lb(1:end-1) = -inf;
    lb(end) = 0;

    ub = zeros(2*length(x_lists.p)+1, 1, 'double');
    ub(1:end-1) = inf;
    ub(end) = 1;

    % First use only bispectrum:
    costgrad = @(x) costgrad_lsqnonlin_projs(x, x_lists, L, M12_quants,...
            M34_quants, m3_micro, m2_micro, m1_micro, lambda(2:end), maxL);

    saveFunc = @(x, optimValues, state) generic_saveFunc(x, optimValues, state, 'bispect', iters_per_save);
    options = optimoptions('lsqnonlin', 'Jacobian', 'on', 'DerivativeCheck', 'off', 'Display', 'iter', 'TolX',1e-4, 'TolFun',1e-4, 'MaxIter', 1e4, 'OutputFcn', saveFunc);

    tic, a_init = lsqnonlin(costgrad, a_init, lb, ub, options); time_bispect = toc
end

gamma = a_init(end);
a_init = a_init(1:end-1);

% Add slice of trispectrum:
costgrad = @(x) costgrad_lsqnonlin_projs_trispect_4tensor(x, x_lists, L, M12_quants,...
    M34_quants, m4_micro, m3_micro, m2_micro, m1_micro, lambda, maxL, gamma, q_cutoff);

saveFunc = @(x, optimValues, state) generic_saveFunc(x, optimValues, state, 'trispect_4tensor', iters_per_save);
options = optimoptions('lsqnonlin', 'Jacobian', 'on', 'DerivativeCheck', 'off', 'Display', 'iter', 'TolX',1e-4, 'TolFun',1e-4, 'MaxIter', 1e4, 'OutputFcn', saveFunc);

tic, x = lsqnonlin(costgrad, a_init, [], [], options); time_TR = toc

a_lms = x(1:end/2) + 1i*x(end/2+1:end);
a_lms = vec_to_cell_vol_coeffs(a_lms, x_lists.L(x_lists.m>=0));
