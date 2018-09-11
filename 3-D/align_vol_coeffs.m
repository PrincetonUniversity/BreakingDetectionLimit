function [a_aligned, R_rec] = align_vol_coeffs(a_lms, a_lms_ref, trials, par_flag)
% Align a volume given in an expansion of spherical harmonics to another. 
%
% Inputs:
%   * a_lms: cell array of expansion coefficients of the volume to be
%   aligned
%   * a_lms_ref: cell array of reference volume
%   * trials: number of random guesses for rotations to align the volumes
%   * par_flag: set to true to try guesses in parallel, and false otherwise
%
% Outputs:
%   * a_aligned: cell array of expansion coefficients of aligned volume
%   * R_rec: 3 x 3 orthogonal matrix aligning a_lms to a_lms_ref
% 
% Eitan Levin, August 2018

problem.M = rotationsfactory(3);
problem.cost = @(Rxyz) cost_rot_alignment(a_lms, a_lms_ref, Rxyz);

R_init = get_init_random_trials(problem, trials, par_flag);
opts.tolgradnorm = 1e-3;
opts.maxiter = 1e2;
[R_rec, cost] = trustregions(problem, R_init, opts);

% Try rotations and a reflection:
J = diag([1;1;-1]);
problem.cost = @(Rxyz) cost_rot_alignment(a_lms, a_lms_ref, J*Rxyz);
R_init = get_init_random_trials(problem, trials, par_flag);
[R_rec_ref, cost_ref] = trustregions(problem, R_init, opts);
if cost_ref < cost
    R_rec = J*R_rec_ref;
    cost = cost_ref;
end

R = RN2RL(getSHrotMtx(R_rec, length(a_lms)-1, 'complex'));
a_aligned = cellfun(@mtimes, a_lms, R, 'UniformOutput', 0);
