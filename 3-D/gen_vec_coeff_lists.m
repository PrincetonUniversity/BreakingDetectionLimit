function [L_list, m_list, s_list, p_inds, n_inds] = gen_vec_coeff_lists(maxL, s_lens)
% Generate list of indices for vecotrized volume expansion coefficients
% a_{l,m,s}.
% 
% Inputs:
%   * maxL: cutoff on order of spherical harmonics
%   * s_lens: list of number of radial frequencies for each spherical
%   harmonics order.
% 
% Outputs:
%   * L_list: list of orders of spherical harmonics corresponding to a
%   given entry in the vector of coefficients.
%   * m_list: list of angular frequencies of spherical harmonics.
%   * s_list: list of radial frequencies.
%   * p_inds: indices of entries corresponding to nonnegative m-indices
%   (used to enforce a real-valued volume)
%   * n_inds: indices of nonpositive m_indices sorted to match those in
%   p_inds.
% 
% Eitan Levin, August 2018

% List the expansion indices (l, m, s)
L_list = []; m_list = []; s_list = [];
for l = 0:maxL
    L_list = [L_list; l*ones(s_lens(l+1)*(2*l+1), 1)];
    m_mat = repmat(-l:l, s_lens(l+1), 1);
    m_list = [m_list; m_mat(:)];
    s_mat = repmat((1:s_lens(l+1)).', 1, 2*l+1);
    s_list = [s_list; s_mat(:)];
end

% List indices corresponding to nonnegative and nonpositive m-indices, used
% to enforce a real-valued volume since then 
% a_{l, -m, s} = (-1)^(l+m) * conj(a_{l, m, s})
p_inds = find(m_list >= 0);
n_inds = zeros(size(p_inds));
for ii = 1:length(p_inds)
    n_inds(ii) = find(L_list == L_list(p_inds(ii)) & m_list == -m_list(p_inds(ii)) & s_list == s_list(p_inds(ii)));
end