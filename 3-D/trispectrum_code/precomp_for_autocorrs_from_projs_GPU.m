function [psi_curr, curr_freqs, psi_lNs, q_list, D_mats, psi_freqs, a_sizes, psi_curr_k0] = ...
                                            precomp_for_autocorrs_from_projs_GPU(maxL, s_list, L, Rots)
% Precomputation for computation of autocorrelations on multiple GPUs.


beta_PSWF = 1; Trunc_img = 1e-5; Trunc_bispect = 1e-1;
[psi_Nn, n_list] = PSWF_2D_full_cart(maxL, L, beta_PSWF, Trunc_img);
psi_Nn = cellfun(@(x) reshape(icfft2(reshape(x, L, L, [])), L^2, []), psi_Nn, 'UniformOutput', 0);

[T, ang_freq] = precomp_shifted_PSWF_coeffs(psi_Nn, n_list, L, beta_PSWF, Trunc_bispect);
num_freqs = length(ang_freq);
q_list = zeros(max(ang_freq) + 1, 1);
for N = 0:max(ang_freq)
    q_list(N+1) = sum(ang_freq == N);
end

a_sizes = s_list; % numel of a_lms
for ii = 0:maxL 
    a_sizes(ii+1) = (2*ii+1)*a_sizes(ii+1); 
end

beta = sph_Bessel_to_2D_PSWF_factors(maxL, n_list, s_list(1), floor(L/2)); % l x N x s x n

psi_lNs = cell(maxL+1, 1);
psi_coeffs_lNs = cell(maxL+1, 1);
for l = 0:maxL
    for N = -l:l
        psi_lNs{l+1}{N+l+1} = (-1)^(N*(N<0))*psi_Nn{N+maxL+1}...
                                                    *beta{l+1}{abs(N)+1}(1:s_list(l+1), :).';
        psi_coeffs_lNs{l+1}{N+l+1} = (-1)^(N*(N<0))*T{N+maxL+1}...
                                                    *beta{l+1}{abs(N)+1}(1:s_list(l+1), :).';
    end
    psi_lNs{l+1} = cat(2, psi_lNs{l+1}{:});
    psi_coeffs_lNs{l+1} = cat(2, psi_coeffs_lNs{l+1}{:});
end

psi_lNs = cat(2, psi_lNs{:});
psi_lNs = mat2cell(psi_lNs, L^2, a_sizes);
psi_lNs = psi_lNs(:);
psi_lNs = cellfun(@(x,y) reshape(x, [], y), psi_lNs, num2cell(a_sizes./s_list), 'UniformOutput', 0);

psi_coeffs_lNs = cat(2, psi_coeffs_lNs{:});
psi_coeffs_lNs = reshape(psi_coeffs_lNs, L^2, num_freqs, []);
psi_coeffs_lNs = mat2cell(psi_coeffs_lNs, L^2, q_list, a_sizes);
psi_coeffs_lNs = permute(psi_coeffs_lNs, [3,2,1]);
psi_coeffs_lNs = cellfun(@(x,y) reshape(x, [], y), psi_coeffs_lNs, repmat(num2cell(a_sizes./s_list), 1, length(q_list)), 'UniformOutput', 0);

psi_coeffs_lNs_k0 = psi_coeffs_lNs(:, 1);

a_sizes = num2cell(a_sizes);

% Rearrange entries of psi_coeffs_lNs for good distribution accross workers
numGPUs = gpuDeviceCount();
if isempty(gcp('nocreate')), parpool(numGPUs); end % numWorkers = numGPUs

psi_freqs = [];
for ii = 1:numGPUs
    psi_freqs = [psi_freqs, ii:numGPUs:length(psi_coeffs_lNs)];
end
psi_coeffs_lNs = distributed(psi_coeffs_lNs(:, psi_freqs));
psi_freqs_dist = distributed(psi_freqs);

% Precompute Wigner-D matrices:
D_mats = cell(maxL + 1, size(Rots, 3));
for ii = 1:size(Rots, 3)
    D_mats(:, ii) = RN2RL(getSHrotMtx(Rots(:,:,ii), maxL, 'complex'));
end
D_mats = cellfun(@transpose, D_mats, 'UniformOutput', 0);

gpuDevice([]); % deselect GPU on client, so all memory is available to workers

spmd
    gd = gpuDevice;
    idx = gd.Index;
    disp(['Using GPU ',num2str(idx)]);
    
    psi_lNs = cellfun(@gpuArray, psi_lNs, 'UniformOutput', 0);
    D_mats = cellfun(@gpuArray, D_mats, 'UniformOutput', 0);
    curr_freqs = getLocalPart(psi_freqs_dist);
    psi_curr = getLocalPart(psi_coeffs_lNs);
    psi_curr_k0 = cellfun(@gpuArray, psi_coeffs_lNs_k0, 'UniformOutput', 0);
end
