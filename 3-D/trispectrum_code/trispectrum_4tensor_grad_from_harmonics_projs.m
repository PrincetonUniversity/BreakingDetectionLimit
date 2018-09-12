function [G, M] = trispectrum_4tensor_grad_from_harmonics_projs(a_lms, psi_curr,...
    psi_curr_k0, curr_freqs, psi_lNs, q_list, D_mats, L, psi_freqs, a_sizes, L_cutoff)
% Function to compute a 4-tensor slice through the trispectrum from a
% number of projections.
% 
% Eitan Levin, August 2018

num_coeffs = length(a_lms);
a_sizes = a_sizes(1:L_cutoff+1, :);

reshape_func = @(x,y) reshape(x, [], y);
% Initialize:
spmd
    a_lms = gpuArray(a_lms);
    psi_lNs = psi_lNs(1:L_cutoff+1, :);
    D_mats = D_mats(1:L_cutoff+1, :);
    psi_curr_k0 = psi_curr_k0(1:L_cutoff+1);

    trials = size(D_mats, 2);

    G = cell(length(curr_freqs), 1);
    M = cell(length(curr_freqs), 1);

    for N_idx = 1:length(curr_freqs)
        N = curr_freqs(N_idx);
        psi_coeffs_curr = cellfun(@gpuArray, psi_curr(1:L_cutoff+1, N_idx), 'UniformOutput', 0);
        G_N = zeros(num_coeffs*q_list(1)*q_list(N), q_list(N), 4, 'gpuArray');     
        G_N3 = zeros(q_list(N)*num_coeffs, q_list(1)*q_list(N), 'gpuArray');
        for ii = 1:trials
            proj_curr_D = cellfun(@mtimes, psi_lNs, D_mats(:, ii), 'UniformOutput', 0);
            proj_curr_D = cellfun(reshape_func, proj_curr_D, a_sizes, 'UniformOutput', 0);
            proj_curr_D = cat(2, proj_curr_D{:}); % L^2 x lms
            proj_curr = proj_curr_D*a_lms; % L^2 x 1

            coeffs_curr_D = cellfun(@mtimes, psi_coeffs_curr, D_mats(:, ii), 'UniformOutput', 0);
            coeffs_curr_D = cellfun(reshape_func, coeffs_curr_D, a_sizes, 'UniformOutput', 0);
            coeffs_curr_D = cat(2, coeffs_curr_D{:}); % (L^2, q) x lms

            coeffs_curr = coeffs_curr_D*a_lms; % (L^2, q) x 1
            coeffs_curr = reshape(coeffs_curr, L^2, q_list(N)); % L^2 x q
            coeffs_curr_D = reshape(coeffs_curr_D, L^2, q_list(N), num_coeffs); % L^2 x q x lms

            coeffs_curr_D_k0 = cellfun(@mtimes, psi_curr_k0, D_mats(:, ii), 'UniformOutput', 0);
            coeffs_curr_D_k0 = cellfun(reshape_func, coeffs_curr_D_k0, a_sizes, 'UniformOutput', 0);
            coeffs_curr_D_k0 = cat(2, coeffs_curr_D_k0{:}); % (L^2, q1) x lms
            coeffs_curr_D_k0 = reshape(coeffs_curr_D_k0, L^2, q_list(1), num_coeffs);
            coeffs_curr_D_k0 = reshape(coeffs_curr_D_k0(:, 1:q_list(1), :), L^2*q_list(1), num_coeffs);

            coeffs_curr_k0 = coeffs_curr_D_k0*a_lms; % (L^2, q1) x 1
            coeffs_curr_k0 = reshape(coeffs_curr_k0, L^2, q_list(1)); % L^2 x q1
            coeffs_curr_D_k0 = reshape(coeffs_curr_D_k0, L^2, q_list(1), num_coeffs); %L^2 x q1 x lms

            tmp = bsxfun(@times, coeffs_curr_k0, permute(coeffs_curr, [1,3,2])); % L^2 x q1 x q2
            tmp2 = bsxfun(@times, tmp, permute(proj_curr_D, [1,3,4,2])); % L^2 x q1 x q2 x lms
            tmp2 = reshape(tmp2, L^2, q_list(1)*q_list(N)*num_coeffs); % L^2 x (q1, q2, lms)
            tmp2 = tmp2.'*conj(coeffs_curr)/L^2/trials; % (q1, q2, lms) x q3
            G_N(:,:,1) = G_N(:,:,1) + tmp2; 
            
            tmp2 = bsxfun(@times, tmp, proj_curr); % L^2 x q1 x q2
            tmp2 = reshape(tmp2, L^2, q_list(1)*q_list(N)); % L^2 x (q1, q2)
            tmp2 = reshape(coeffs_curr_D, L^2, q_list(N)*num_coeffs)'*tmp2/L^2/trials; % (q3, lms) x (q1, q2);
            G_N3 = G_N3 + tmp2;

            tmp = bsxfun(@times, permute(coeffs_curr_D, [1,4,2,3]), coeffs_curr_k0); % L^2 x q1 x q2 x lms
            tmp = bsxfun(@times, reshape(tmp, L^2, q_list(1)*q_list(N)*num_coeffs), proj_curr);
            G_N(:,:,2) = G_N(:,:,2) + tmp.'*conj(coeffs_curr)/L^2/trials; % (q1, q2, lms) x q3

            tmp = bsxfun(@times, permute(coeffs_curr_D_k0, [1,4,2,3]), coeffs_curr); % L^2 x q2 x q1 x lms
            tmp = permute(tmp, [1,3,2,4]); % L^2 x q1 x q2 x lms
            tmp = bsxfun(@times, reshape(tmp, L^2, q_list(1)*q_list(N)*num_coeffs), proj_curr);
            G_N(:,:,3) = G_N(:,:,3) + tmp.'*conj(coeffs_curr)/L^2/trials; % (q1, q2, lms) x q3
        end
        G_N3 = reshape(G_N3, q_list(N), num_coeffs, q_list(1), q_list(N)); % q3 x lms x q1 x q2
        G_N3 = conj(permute(G_N3, [3,4,2,1])); % q1 x q2 x lms x q3
        G_N(:,:,4) = reshape(G_N3, q_list(1)*q_list(N)*num_coeffs, q_list(N));

        % permute to factor lms out:
        G_N = permute(reshape(G_N, q_list(1), q_list(N), num_coeffs, q_list(N), 4), [3,1,2,4,5]); % lms x q1 x q2 x q3 x 4

        G_N = reshape(G_N, num_coeffs, q_list(1)*q_list(N)^2, 4); % lms x (q1, q2, q3) x 4

        M{N_idx} = real(G_N(:,:,4).'*a_lms);
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

