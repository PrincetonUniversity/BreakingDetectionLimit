function [G, M] = trispectrum_grad_from_harmonics_projs...
    (a_lms, psi_coeffs_lNs, psi_lNs, q_list, D_mats, L, k_max)
% Compute the trispectrum and its gradient with respect to the volume
% expansion coefficients, by averaging the trispectrum of a number of
% projections.
% 
% Inputs:
%   * a_lms: cell array of volume expansion coefficients
%   * psi_coeffs_lNs, psi_lNs: inner products of linear combinations of
%   shifted PSWFs with centered ones, and the linear combinations
%   themselves (see precomp_for_autocorrs_from_projs.m)
%   * q_list: list of number of radial frequencies per angular frequency of
%   PSWFs (see precomp_for_autocorrs_from_projs.m)
%   * D_mats: Wigner-D matrices of viewing directions used to estimate the
%   trispectrum
%   * L: length of volume or projection
%   * k_max: maximum angular frequency for the trispectrum computation (to
%   compute only a subset of the full trispectrum). Leave empty to use
%   maximum number (dictated by cutoffs of PSWF expansion).
% 
% Outputs:
%   * G: gradient of the trispectrum with respect to a_lms
%   * M: the trispectrum
% 
% Eitan Levin, August 2018

a_vec = vec_cell(a_lms);
num_coeffs = length(a_vec);
maxK = (length(q_list) - 1)/2;

if ~exist('k_max', 'var') || isempty(k_max)
    k_max = maxK;
end

trials = size(D_mats, 2);

G = 0;
M = 0;

parfor ii = 1:trials % parallelize on projections - ideally modify to use GPUs
    G_curr = cell(2*k_max+1, 2*k_max+1);
    G_curr = cellfun(@(x) 0, G_curr, 'UniformOutput', 0);
    M_curr = cell(2*k_max+1, 2*k_max+1);

    proj_curr_D = cellfun(@mtimes, psi_lNs, D_mats(:, ii), 'UniformOutput', 0);
    proj_curr_D = cellfun(@(x,y) reshape(x, [] ,numel(y)), proj_curr_D, a_lms, 'UniformOutput', 0);
    proj_curr_D = cat(2, proj_curr_D{:}); % L^2 x lms
    proj_curr = proj_curr_D*a_vec; % L^2 x 1

    coeffs_curr_D = cell(4*k_max+1,1);
    coeffs_curr = cell(4*k_max+1,1);
    for k = -2*k_max:2*k_max
        T = cellfun(@mtimes, psi_coeffs_lNs(:,k+2*k_max+1), D_mats(:, ii), 'UniformOutput', 0);
        T = cellfun(@(x,y) reshape(x, [], numel(y)), T, a_lms, 'UniformOutput', 0);
        T2 = cellfun(@(x,y) x*y(:), T, a_lms, 'UniformOutput', 0);
        coeffs_curr{k+2*k_max+1} = reshape(sum(cat(2, T2{:}), 2), L^2, q_list(k+maxK+1));
        coeffs_curr_D{k+2*k_max+1} = reshape(cat(2,T{:}), L^2, q_list(k+maxK+1), num_coeffs); % L^2 x q x lms
    end

    for k1 = -k_max : k_max
        q1 = q_list(k1+maxK+1);

        C_k1 = coeffs_curr{k1+2*k_max+1};
        C_D_k1 = coeffs_curr_D{k1+2*k_max+1};
        for k2 = -k_max : k_max
            Gt = cell(4,1);
            Gt = cellfun(@(x) 0, Gt, 'UniformOutput', 0);

            q2 = q_list(k2+maxK+1);
            C_k2 = coeffs_curr{k2+2*k_max+1};
            C_D_k2 = coeffs_curr_D{k2+2*k_max+1};

            k3 = k1 + k2;
            q3 = q_list(k3+maxK+1);
            C_k3 = coeffs_curr{k3+2*k_max+1};
            C_D_k3 = coeffs_curr_D{k3+2*k_max+1};
            
            tmp = bsxfun(@times, C_k1, permute(C_k2, [1,3,2])); % L^2 x q1 x q2
            tmp2 = bsxfun(@times, tmp, permute(proj_curr_D, [1,3,4,2])); % L^2 x q1 x q2 x lms
            tmp2 = reshape(tmp2, L^2, q1*q2*num_coeffs); % L^2 x (q1, q2, lms)
            tmp2 = tmp2.'*conj(C_k3)/L^2/trials; % (q1, q2, lms) x q3
            
            Gt{1} = Gt{1} + tmp2;
                    
            tmp2 = bsxfun(@times, tmp, proj_curr); % L^2 x q1 x q2
            tmp2 = reshape(tmp2, L^2, q1*q2); % L^2 x (q1, q2)
            tmp2 = reshape(C_D_k3, L^2, q3*num_coeffs)'*tmp2/L^2/trials; % (q3, lms) x (q1, q2);
            tmp2 = reshape(tmp2, q3, num_coeffs, q1, q2); % q3 x lms x q1 x q2
            tmp2 = conj(permute(tmp2, [3,4,2,1])); % q1 x q2 x lms x q3
            tmp2 = reshape(tmp2, q1*q2*num_coeffs, q3);
            Gt{2} = Gt{2} + tmp2;

            tmp = bsxfun(@times, permute(C_D_k2, [1,4,2,3]), C_k1); % L^2 x q1 x q2 x lms
            tmp = bsxfun(@times, reshape(tmp, L^2, q1*q2*num_coeffs), proj_curr);
            tmp = tmp.'*conj(C_k3)/L^2/trials; % (q1, q2, lms) x q3
            
            Gt{3} = Gt{3} + tmp;

            tmp = bsxfun(@times, permute(C_D_k1, [1,4,2,3]), C_k2); % L^2 x q2 x q1 x lms
            tmp = permute(tmp, [1,3,2,4]); % L^2 x q1 x q2 x lms
            tmp = bsxfun(@times, reshape(tmp, L^2, q1*q2*num_coeffs), proj_curr);
            tmp = tmp.'*conj(C_k3)/L^2/trials; % (q1, q2, lms) x q3
            
            Gt{4} = Gt{4} + tmp;

            Gt = cellfun(@(x) ...
    reshape(permute(reshape(x, q1, q2, num_coeffs, q3), [3,1,2,4]), num_coeffs, q1*q2*q3),...
            Gt, 'UniformOutput', 0); % lms x q1 x q2 x q3 x 4

            M_curr{k1+k_max+1, k2+k_max+1} = reshape(real(Gt{1}.'*a_vec), q1, q2, q3);

            G_curr{k1+k_max+1, k2+k_max+1} = Gt{1} + Gt{2} + Gt{3} + Gt{4};
        end
    end

    G = G + cat(2, G_curr{:});
    M = M + vec_cell(M_curr);
end
