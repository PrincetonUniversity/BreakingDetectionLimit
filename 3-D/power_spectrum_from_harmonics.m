function [M, G, H] = power_spectrum_from_harmonics(a_vec, j_l, R_0n, w, L, maxL)
% Evaluate the power spectrum from volume expansion coefficients.
% See reconstruct_from_clean_autocorrs_script.m for example usage.
% 
% Inputs:
%   * a_vec: vecotrized volume expansion coefficients
%   * j_l: cell array of spherical bessel functions (see
%   generate_spherical_bessel_basis.m)
%   * R_0n: matrix of zeroth order radial PSWFs (see PSWF_radial_2D.m)
%   * w: quadrature weights for numerical integration
%   * L: length of volume (or projection)
%   * maxL: spherical harmonic expansion cutoff
% 
% Outputs:
%   * M: expansion coefficients of the power spectrum in 2D PSWFs.
%   * G: gradient of M with respect to a_vec
%   * H: Hessian of M with respect to a_vec

H = cell(size(R_0n,2),1);
H = cellfun(@(x) cell(maxL+1,1), H, 'UniformOutput', 0);
for ll = 0:maxL
    JJ = bsxfun(@times, j_l{ll+1}, permute(j_l{ll+1}, [1,3,2])); % r x s1 x s2
    JJ = reshape(JJ, size(JJ,1), []); % r x (s1, s2)
    JJ = sqrt(pi/2)/(4*pi*L^2)*JJ.'*diag(w)*R_0n; % (s1, s2) x n
    for n = 1:size(JJ,2)
        H{n}{ll+1} = kron(eye(2*ll+1), reshape(JJ(:, n), size(j_l{ll+1},2), size(j_l{ll+1},2)));
    end
end
H = cellfun(@(x) blkdiag(x{:}), H, 'UniformOutput', 0);

G = cellfun(@(x) x*a_vec, H, 'UniformOutput', 0);
M = cell2mat(cellfun(@(x) a_vec'*x, G, 'UniformOutput', 0));

G = cat(2,G{:});
