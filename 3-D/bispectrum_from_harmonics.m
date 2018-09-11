function M = bispectrum_from_harmonics(a_lms, B, W, L1_list, L2_list, L3_list, s_lens, blk_id)
% Compute the bispectrum for the volume expansion coefficients a_lms. Uses
% precomputations done in precomp_B_factors_script.m.
% 
% Inputs:
%   * a_lms: cell array of volume expansion coefficients
%   * B: precomputed factors for bispectrum evaluation (see
%   precomp_bispectrum_coeffs_matMult.m)
%   * W: precomputed weights involving Wigner 3j symbols (see
%   precomp_wigner_weights.m)
%   * L1_list, L2_list, L3_list: lists of indices for vectorization in B
%   (see precomp_bispectrum_coeffs_matMult.m)
%   * s_lens: number of radial frequencies for vectorization of volume
%   expansion coefficients (see gen_vec_coeff_lists.m)
%   * blk_id: list of angular frequencies for vectorization of bispectrum
%   (see precomp_bispectrum_coeffs_matMult.m)
% 
% Outputs:
%   * M: bispectrum (vectorized by default, uncomment code block at the end
%   of function to obtain a cell array sorted by angular frequency)
% 
% Eitan Levin, August 2018

kq_len = length(blk_id);
M(kq_len, 1) = 0;
parfor ii = 1:length(B)
    L1 = L1_list(ii); s1_curr = s_lens(L1+1);
    L2 = L2_list(ii); s2_curr = s_lens(L2+1);
    jj_3 = L3_list(ii); L3 = abs(L1-L2) + jj_3 - 1; s3_curr = s_lens(L3+1);
    
    % Perform summation over s1, s2, s3:
    tmp = reshape(B{ii}(:,1:s3_curr,1:s2_curr,1:s1_curr), kq_len*s3_curr*s2_curr, s1_curr)*a_lms{L1+1}; % ((k,q1,q2), s3, s2) x m1
    
    tmp = reshape(tmp.', (2*L1+1)*kq_len*s3_curr, s2_curr)*a_lms{L2+1}; % (m1, (k,q1,q2), s3) x m2
    
    tmp = reshape(tmp.', (2*L2+1)*(2*L1+1)*kq_len, s3_curr)*conj(a_lms{L3+1}); % (m2, m1, (k,q1,q2)) x m3
    
    tmp = reshape(tmp.', 2*L3+1, 2*L2+1, 2*L1+1, kq_len); % m3 x m2 x m1 x (k,q1,q2)
    
    % Perform summation over m1, m2:
    for m1 = -L1:L1
        for m2 = max(-L2, -L3-m1):min(L2, L3-m1)
            m3 = m1+m2;
            M = M + W{L1+1, L2+1}(m2+L2+1, m1+L1+1, jj_3)*reshape(tmp(m3+L3+1, m2+L2+1, m1+L1+1,:), kq_len, 1); % (k,q1,q2)
        end
    end
end

% % Not needed for optimization, only for debugging:
% M = accumarray(blk_id, M, [], @(x) {x});
% M = cellfun(@(x) reshape(x, sqrt(length(x)), sqrt(length(x))), M, 'UniformOutput', 0);
% M = cellfun(@(x) real(x+x')/2, M, 'UniformOutput', 0);
