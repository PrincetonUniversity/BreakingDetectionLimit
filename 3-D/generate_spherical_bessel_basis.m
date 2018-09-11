function j_l = generate_spherical_bessel_basis(maxL, s_list, k_max, k_grid)
% Generate spherical Bessel functions used for expansion of the volume
% 
% Inputs:
%   * maxL: cutoff for spherical harmonics expansion
%   * s_list: list of number of radial frequencies per spherical harmonics
%   order
%   * k_max: assumed bandlimit
%   * k_grid: radial points for evaluation
% 
% Outputs:
%   * j_l: cell array of spherical Bessel functions, sorted by spherical
%   harmonics order
% 
% Eitan Levin, June 2018 

k_grid = k_grid(:);
j_l = cell(maxL+1,1);
for l = 0:maxL
    r_li = zerobess('J', l+1/2, s_list(l+1)); % fast zeroes of bessel functions
    j_lf = @(x,l) sqrt(pi./(2*x)).*besselj(l+1/2, x);
    X = k_grid*r_li.'/k_max;
    Y = ones(size(k_grid))*r_li.';
    J_vals = j_lf(X, l);
    if l > 0
        J_vals(isnan(J_vals)) = 0;
    else
        J_vals(isnan(J_vals)) = 1;
    end
    J_vals(k_grid > k_max) = 0; % enforce support
    j_l{l+1} = J_vals.*sqrt(2)./(k_max^(3/2)*abs( j_lf(Y, l+1) ));
    
end