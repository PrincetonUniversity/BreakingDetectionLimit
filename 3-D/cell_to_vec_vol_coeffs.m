function a_lms_vec = cell_to_vec_vol_coeffs(a_lms)
% Takes a cell array of all volume expansion coefficients a_{l,m,s} 
% (generated as in e.g. reconstruct_from_clean_autocorrs_script.m), take
% only coefficients with m >= 0 which uniquely determine a real-valued
% volume, and vectorize the resulting cell-array.
% 
% Inputs:
%   * a_lms: cell array of volume expansion coefficients (see e.g.
%   reconstruct_from_clean_autocorrs_script.m)
% 
% Outputs:
%   * a_lms_vec: vector of expansion coefficients with m >= 0
% 
% Eitan Levin, July 2018

a_lms_vec = cellfun(@(x) x(:, (size(x,2)-1)/2+1:end), a_lms, 'UniformOutput', 0);
a_lms_vec = vec_cell(a_lms_vec);