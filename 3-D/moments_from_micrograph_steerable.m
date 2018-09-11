function [m1, m2, m3] = moments_from_micrograph_steerable(I, L, patch_len)
% Function to compute the first three autocorrelations of the micrographs,
% accounting for all of its in-plane rotations. 
% 
% Inputs:
%   * I: Either a 2D array representing the micrograph, or a 3D array
%   representing a stack of micrographs in which case they are concatenated
%   along the 2nd dimension.
%   * L: length of volume (or projection), so the support of the
%   autocorrelaions are 2L-1
%   * patch_len: number of rows and columns for each patch, before padding by
%   2(L-1) to compute aperiodic rather than periodic correlation
% 
% Eitan Levin, July 2018

% L is size of single projection
m1 = mean(I(:))/L;

beta = 1;       % Bandlimit ratio (between 0 and 1) - smaller values stand for greater oversampling
T = 1e-1;       % Truncation parameter

[Wt, ang_freq] = precomp_pswf_t_windows(2*L-1, beta, T);

maxN = max(ang_freq);
q_list = zeros(maxN+1, 1);
for ii = 0:maxN
    q_list(ii+1) = sum(ang_freq == ii); 
end
num_freqs = length(ang_freq);
q_cumsum = cumsum([0; q_list(:)]);

% Setup multiple GPUs:
gpuDevice([]); % deselect GPU on client, so all memory is available to workers
if isempty(gcp('nocreate')), parpool(gpuDeviceCount()); end % numWorkers = numGPUs

spmd
    gd = gpuDevice;
    idx = gd.Index;
    disp(['Using GPU ',num2str(idx)]);
    Wt = gpuArray(Wt);

    m2 = zeros(num_freqs, 1, 'gpuArray');
    m3 = zeros(num_freqs, num_freqs, 'gpuArray'); 
end

for ii = 1:size(I, 3)
    [megaPatches, cntrs, sz_patches] = micrograph_to_patches(I(:,:,ii), patch_len, patch_len, L);

    megaPatches = distributed(megaPatches);
    cntrs = distributed(cntrs);
    sz_patches = distributed(sz_patches);

    spmd    
        patches_curr = getLocalPart(megaPatches);
        cntrs_curr = getLocalPart(cntrs);
        sizes_curr = getLocalPart(sz_patches);
        W_curr = fft2(Wt, sizes_curr{1}(1)+2*(L-1), sizes_curr{1}(2)+2*(L-1));
        for t = 1:numel(patches_curr)
            I_curr = gpuArray(patches_curr{t});
            cntr_curr = gpuArray(cntrs_curr{t});
            sz_curr = sizes_curr{t};
            
            if t == 1 || any(sz_curr ~= sizes_curr{t-1})
                W_curr = fft2(Wt, sz_curr(1)+2*(L-1), sz_curr(2)+2*(L-1));
            end

            coeffs = ifft2(bsxfun(@times, W_curr, I_curr));
            coeffs = coeffs(2*L-1:sz_curr(1), 2*L-1:sz_curr(2), :);
            coeffs = reshape(coeffs, [], size(coeffs, 3));

            m2 = m2 + coeffs.'*cntr_curr(:);
   
            m3_add = bsxfun(@times, coeffs, cntr_curr(:));
            m3_add = real(m3_add'*coeffs);
            m3 = m3 + m3_add;
        end
    end
end

spmd
    m2 = gather(m2);
    m3 = gather(m3);
end

m2 = sum(cat(2, m2{:}), 2);
m3 = sum(cat(3, m3{:}), 3);

norm_factor = (size(I,1)-2*(L-1))*(size(I,2)-2*(L-1))*size(I,3);
m2 = m2./norm_factor;
m3 = m3./norm_factor;

m2 = m2(1:q_list(1));
m3_cell = cell(size(q_list));
for ii = 1:length(q_list)
    m3_cell{ii} = m3(q_cumsum(ii)+1:q_cumsum(ii+1), q_cumsum(ii)+1:q_cumsum(ii+1));
end
m3 = m3_cell;
