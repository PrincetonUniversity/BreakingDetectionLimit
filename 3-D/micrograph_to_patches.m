function [megaPatches, cntrs, sz_patches] = micrograph_to_patches(I, rows_per_patch, cols_per_patch, L)
% Split a micrograph into patches with overlaps so that aperiodic correlations
% between the micrograph and a small window is given by appropriately
% concatenating the aperiodic correlations with the patches. Used to expand 
% patches in the micrograph in PSWFs.
% 
% Inputs:
%   * I: Either a 2D array representing the micrograph, or a 3D array
%   representing a stack of micrographs in which case they are concatenated
%   along the 2nd dimension.
%   * rows_per_patch: number of rows for each patch, before padding by
%   2(L-1) to compute aperiodic rather than periodic correlation
%   * cols_per_patch: number of rows for each patch before padding as above
%   * L: length of volume (or projection), so the support of the
%   autocorrelaions are 2L-1
% 
% Eitan Levin, July 2018

I = reshape(I, size(I,1), []); % concatenate micrographs along columns
num_row_patches = ceil(size(I,1)/(rows_per_patch-2*L+2));
num_col_patches = ceil(size(I,2)/(cols_per_patch-2*L+2));

megaPatches = cell(num_row_patches, num_col_patches);
megaPatches{1, 1} = I(1:rows_per_patch, 1:cols_per_patch);

for jj = 1:num_col_patches
    inds_col = (jj-1)*(cols_per_patch-2*L+2)+1:min(jj*cols_per_patch-(jj-1)*(2*L-2), size(I,2));
    for ii = 1:num_row_patches
        inds_row = (ii-1)*(rows_per_patch-2*L+2)+1:min(ii*rows_per_patch-(ii-1)*(2*L-2), size(I,1));

        megaPatches{ii, jj} = I(inds_row, inds_col);
    end
end

%vec = @(x)x(:);
%megaPatches = [vec(megaPatches(1:end-1, 1:end-1)); megaPatches(:,end);vec(megaPatches(end,1:end-1))];
sz_patches = cellfun(@size, megaPatches, 'UniformOutput', 0);

cntrs = cell(size(megaPatches));
for ii = 1:numel(megaPatches)
    cntrs{ii} = megaPatches{ii}(L:end-L+1, L:end-L+1);
    megaPatches{ii} = fft2(megaPatches{ii}, sz_patches{ii}(1)+2*(L-1), sz_patches{ii}(2)+2*(L-1));
end

inds = cellfun(@isempty, cntrs);
cntrs(inds) = [];
megaPatches(inds) = [];
sz_patches(inds) = [];
