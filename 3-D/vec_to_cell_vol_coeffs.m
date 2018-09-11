function a_lms = vec_to_cell_vol_coeffs(a_lms_vec, L_list)
% Convert a vector of volume expansion coefficients a_{l,m,s} with m >= 0 
% to a cell array with -l <= m <= l. 
% (See also cell_to_vec_vol_coeffs.m)
% 
% Inputs:
%   * a_lms_vec: vector of coefficients a_{l,m,s} with m >= 0, (see 
%   cell_to_vec_vol_coeffs.m)
%   * L_list: list of orders of spherical harmonics corresponding to each
%   entry in a_lms_vec (see e.g. recover_vol_coeffs_from_moments.m)
% 
% Outputs:
%   * a_lms: cell array of all volume expansion coefficients (see e.g.
%   reconstruct_from_clean_autocorrs_script.m)
% 
% Eitan Levin, July 2018

a_lms = accumarray(L_list+1, a_lms_vec, [max(L_list)+1, 1], @(x) {x});
for l = 0:length(a_lms)-1
    a_lms{l+1} = reshape(a_lms{l+1}, [], l+1);
    a_lms{l+1}(:,1) = (a_lms{l+1}(:,1) + (-1)^l*conj(a_lms{l+1}(:,1)))/2;
    neg_m = bsxfun(@times, conj(a_lms{l+1}(:,end:-1:2)), (-1).^(l+(l:-1:1)));
    a_lms{l+1} = [neg_m, a_lms{l+1}];
end