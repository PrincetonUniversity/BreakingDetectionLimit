function [G, M] = bispectrum_grad_from_harmonics_projs(a_lms, psi_curr, curr_freqs, psi_lNs, q_list, D_mats, L, psi_freqs, a_sizes, L_cutoff)

num_coeffs = length(a_lms);
a_sizes = a_sizes(1:L_cutoff+1, :);

reshape_func = @(x,y) reshape(x, [], y);
spmd
    a_lms = gpuArray(a_lms);
    psi_lNs = psi_lNs(1:L_cutoff+1, :);
    D_mats = D_mats(1:L_cutoff+1, :);

    trials = size(D_mats, 2);
    G = cell(length(curr_freqs), 1);
    M = cell(length(curr_freqs), 1);
    for N_idx = 1:length(curr_freqs)
        N = curr_freqs(N_idx);
        psi_coeffs_curr = cellfun(@gpuArray, psi_curr(1:L_cutoff+1, N_idx), 'UniformOutput', 0);

        G_N = zeros(num_coeffs*q_list(N), q_list(N), 3, 'gpuArray');     
        for ii = 1:trials
            proj_curr_D = cellfun(@mtimes, psi_lNs, D_mats(:, ii), 'UniformOutput', 0);
            proj_curr_D = cellfun(reshape_func, proj_curr_D, a_sizes, 'UniformOutput', 0);
            proj_curr_D = cat(2, proj_curr_D{:}); % L^2 x lms
            proj_curr = proj_curr_D*a_lms; % L^2 x 1

            proj_curr_D = permute(proj_curr_D, [1,3,2]); % L^2 x 1 x lms

            coeffs_curr_D = cellfun(@mtimes, psi_coeffs_curr, D_mats(:, ii), 'UniformOutput', 0);
            coeffs_curr_D = cellfun(reshape_func, coeffs_curr_D, a_sizes, 'UniformOutput', 0);
            coeffs_curr_D = cat(2, coeffs_curr_D{:}); % (L^2, q) x lms

            coeffs_curr = coeffs_curr_D*a_lms; % (L^2, q) x 1

            coeffs_curr_D = reshape(coeffs_curr_D, L^2, q_list(N)*num_coeffs); % L^2 x (q, lms)
            coeffs_curr = reshape(coeffs_curr, L^2, q_list(N)); % L^2 x q
            
            tmp = bsxfun(@times, coeffs_curr, proj_curr_D); % L^2 x q x lms
            tmp = reshape(tmp, L^2, q_list(N)*num_coeffs); % L^2 x (q, lms)

            G_N(:,:,1) = G_N(:,:,1) + tmp.'*conj(coeffs_curr)/L^2/trials; % (q, lms) x q

            tmp = bsxfun(@times, coeffs_curr_D, proj_curr); % L^2 x (q, lms)
            G_N(:,:,2) = G_N(:,:,2) + tmp.'*conj(coeffs_curr)/L^2/trials; % (q, lms) x q
        end
        
        % permute to factor lms out:
        G_N = permute(reshape(G_N, q_list(N), num_coeffs, q_list(N), 3), [2,1,3,4]); % lms x q1 x q2
        G_N(:,:,:,3) = permute(G_N(:,:,:,2), [1, 3, 2]);

        G_N = reshape(G_N, num_coeffs, q_list(N)^2, 3); % lms x (q1, q2) x 3

        M{N_idx} = real(G_N(:,:,1).'*a_lms);
        G{N_idx} = sum(G_N, 3);
    end
    G = cellfun(@gather, G, 'UniformOutput', 0);
    M = cellfun(@gather, M, 'UniformOutput', 0);
end

G = cat(1, G{:}); 
M = cat(1, M{:});

% invert permutation:
G(psi_freqs) = G; 
M(psi_freqs) = M;

G = cat(2, G{:});

