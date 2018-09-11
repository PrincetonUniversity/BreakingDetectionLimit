function beta = sph_Bessel_to_2D_PSWF_factors(maxL, n_list, maxS, L)
% Compute factors proportioanl to the integral of 2D PSWFs and spherical
% Bessel functions, used to evaluate the bispectrum.
% 
% Inputs: 
%   * maxL: cutoff for spherical harmonics expansion
%   * n_list: list of number of radial frequencies for each angular
%   frequency of 2D PSWFs
%   * maxS: maximum radial frequency for spherical Bessel functions
%   * L: length of volume or projection
% 
% Outputs:
%   * beta: a cell array indexed (order of spherical harmonics) x (angular
%   frequency for 2D PSWFs) containing the factors.
% 
% Eitan Levin, June 2018

c = pi*L;
[r,w]=lgwt(200*n_list(1),0,1);
w = w.*r;

Y_l = YN2YL(getSH(maxL, [0, pi/2], 'complex'));
j_l = generate_spherical_bessel_basis(maxL, maxS*ones(maxL+1,1), 1/2, r/2);

beta = cell(maxL+1,1);
for l = 0:maxL
    beta{l+1} = cell(l+1, 1);
    for N = 0:l % only compute for positive N
        [R_Nn, alpha_Nn_2D] = PSWF_radial_2D(abs(N), n_list(abs(N)+1)-1, c, r); % generate 2D radial prolates
%        R_Nn = bsxfun(@times, R_Nn, 1./alpha_Nn_2D.');
        R_Nn = bsxfun(@times, R_Nn, 4./alpha_Nn_2D(:).^2.');
        j_l_curr = sqrt(2*pi)*Y_l{l+1}(l+1+N)*j_l{l+1};
        
        beta{l+1}{N+1} = j_l_curr.'*diag(w)*R_Nn; % s x n
    end
end
