function [G1, G2, G3] = bispectrum_grad_from_harmonics(a_lms, B, W, L1_list, L2_list, L3_list, L_list, m_list, s_lens, blk_id)
% Compute the three terms in the gradient of the bispectrum. Uses
% precomputations done in precomp_B_factors_script.m.
% See reconstruct_from_clean_autocorrs_script.m for example usage.
% 
% Inputs:
%   * a_lms: cell array of volume expansion coefficients
%   * B: precomputed factors for bispectrum evaluation (see
%   precomp_bispectrum_coeffs_matMult.m)
%   * W: precomputed weights involving Wigner 3j symbols (see
%   precomp_wigner_weights.m)
%   * L1_list, L2_list, L3_list: lists of indices for vectorization in B
%   (see precomp_bispectrum_coeffs_matMult.m)
%   * L_list, m_list, s_lens: lists of indices for vectorization of volume
%   expansion coefficients (see gen_vec_coeff_lists.m)
%   * blk_id: list of angular frequencies for vectorization of bispectrum
%   (see precomp_bispectrum_coeffs_matMult.m)
% 
% Outputs:
%   * G1, G2, G3: the three terms on the gradient of the bispectrum with
%   respect to the coefficients of the volume. 
% 
% Eitan Levin, August 2018

kq_len = length(blk_id); % number of elements in the bispectrum
% Initialize:
G1(length(L_list), kq_len) = 0; 
G2(length(L_list), kq_len) = 0; 
G3(length(L_list), kq_len) = 0;
parfor ii = 1:length(B)
    L1 = L1_list(ii); s1_curr = s_lens(L1+1);
    L2 = L2_list(ii); s2_curr = s_lens(L2+1);
    jj_3 = L3_list(ii); L3 = abs(L1-L2) + jj_3 - 1; s3_curr = s_lens(L3+1);
    
    % Sum over s2, s3 (re-using tmp_1 to decrease memory and allocation costs):
    tmp_1 = permute(B{ii}(:, 1:s3_curr, 1:s2_curr, 1:s1_curr), [4,1,2,3]); % s1 x (k,q1,q2) x s3 x s2
    tmp_1 = reshape(tmp_1, s1_curr*kq_len*s3_curr, s2_curr)*a_lms{L2+1}; % (s1, (k,q1,q2), s3) x m2
    tmp_1 = reshape(tmp_1.', (2*L2+1)*s1_curr*kq_len, s3_curr)*conj(a_lms{L3+1}); % (m2, s1, (k,q1,q2)) x m3
    tmp_1 = reshape(tmp_1.', 2*L3+1, 2*L2+1, s1_curr, kq_len); % m3 x m2 x s1 x (k,q1,q2)
    
    % Sum over m1, m2:
    g1 = zeros(length(L_list), kq_len);
    L1_inds = (L_list == L1);
    for m1 = -L1:L1
        m1_inds = L1_inds & (m_list == m1);
        for m2 = max(-L2, -L3-m1):min(L2, L3-m1)
            m3 = m1+m2;
            g1(m1_inds, :) = g1(m1_inds, :)...
                + W{L1+1,L2+1}(m2+L2+1, m1+L1+1, jj_3)*reshape(tmp_1(m3+L3+1, m2+L2+1, :,:), s1_curr, kq_len); % s1 x q1 x q2
        end
    end
    
    % Sum over s1, s2, s3
    tmp_1 = reshape(B{ii}(:, 1:s3_curr, 1:s2_curr, 1:s1_curr), kq_len*s3_curr*s2_curr, s1_curr)*a_lms{L1+1}; % ((k,q1,q2), s3, s2) x m1
    tmp_1 = reshape(tmp_1.', (2*L1+1)*kq_len*s3_curr, s2_curr); % (m1, (k,q1,q2), s3) x s2
    
    tmp_2 = tmp_1.'; % s2 x (m1, (k,q1,q2), s3)
    tmp_2 = reshape(tmp_2, s2_curr*(2*L1+1)*kq_len, s3_curr)*conj(a_lms{L3+1}); % (s2, m1, (k,q1,q2)) x m3
    tmp_2 = reshape(tmp_2.', 2*L3+1, s2_curr, 2*L1+1, kq_len); % m3 x s2 x m1 x (k,q1,q2)
    
    tmp_1 = tmp_1*a_lms{L2+1}; % (m1, (k,q1,q2), s3) x m2
    tmp_1 = reshape(tmp_1.', 2*L2+1, 2*L1+1, kq_len, s3_curr); % m2 x m1 x (k,q1,q2) x s3
    
    % Sum over m1, m2:
    g2 = zeros(length(L_list), kq_len);
    g3 = zeros(length(L_list), kq_len);
    
    L2_inds = (L_list == L2);
    L3_inds = (L_list == L3);
    for m1 = -L1:L1
        for m2 = max(-L2, -L3-m1):min(L2, L3-m1)
            m3 = m1+m2;
            norm_fact = W{L1+1,L2+1}(m2+L2+1, m1+L1+1, jj_3);
            
            g2(L2_inds & (m_list == m2), :) = g2(L2_inds & (m_list == m2), :)...
                + norm_fact*reshape(tmp_2(m3+L3+1, :, m1+L1+1, :), s2_curr, kq_len); % s2 x q1 x q2
            
            g3(L3_inds & (m_list == m3), :) = g3(L3_inds & (m_list == m3), :)...
                + norm_fact*reshape(tmp_1(m2+L2+1, m1+L1+1, :,:), kq_len, s3_curr).'; % s3 x q1 x q2
        end
    end
    G1 = G1 + g1;
    G2 = G2 + g2;
    G3 = G3 + g3;
end

