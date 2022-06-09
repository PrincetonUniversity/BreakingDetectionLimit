info.mol = 'NAME OF MOLECULE';
vol = double(ReadMRC([info.mol '.mrc'])); %read volume

L = 20; % size of vol after downsampling
vol = cryo_downsample(vol, L);

M = 7420; % size of a micrograph
num_micros = 300; % number of micrographs

info.maxL = 5; % L_max for volume
info.L0 = size(vol,1);
info.r_cut = 1/2; % Nyquist
info.N = floor(info.L0/2);

[Psilms, Psilms_2D, jball, jball_2D] = precompute_spherical_basis(info);
a_lms = expand_vol_spherical_basis(vol, info, Psilms, jball); % expand volume in 3D steerable basis

I = zeros(M, M, num_micros, 'double'); % micrographs
gamma = zeros(num_micros, 1, 'double'); % occupancy factors

if isempty(gcp('nocreate')), parpool('local', maxNumCompThreads); end

parfor t = 1:num_micros % for each micrograph
    num_projs = 0; % counter for projections
    inds_possible = cart2inds([M,M], 1:M-L+1, 1:M-L+1); % possible indices for upper-left corner
    curr_micro = zeros(M,M,'double');
    while ~isempty(inds_possible)
        % Put projection at random possible index, increment counter:
        [off_1, off_2] = ind2sub([M,M],inds_possible(randi(length(inds_possible))));

        projs_curr = gen_rand_proj(a_lms, Psilms_2D, jball_2D, L);
        curr_micro(off_1:off_1+L-1, off_2:off_2+L-1) = projs_curr;        
        num_projs = num_projs + 1;
        
        % Delete indices near current projection from possible indices:
        inds_to_del = cart2inds([M,M], ...
            off_1-2*(L-1):off_1+2*(L-1), off_2-2*(L-1):off_2+2*(L-1));
        inds_possible = setdiff(inds_possible, inds_to_del);
    end
    I(:,:,t) = curr_micro;
    gamma(t) = num_projs*L^2/M^2;
end

save(['./' info.mol '_micros_' num2str(num_micros) '_M_' num2str(M) '_maxL_' num2str(info.maxL) '.mat'], '-v7.3')
