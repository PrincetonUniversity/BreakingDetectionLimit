clear;
close all;
clc;

% arbitrary seed
rng(956546);

%% Pick parameters and generate signals
if isempty(gcp('nocreate'))
    parpool(72,'IdleTimeout', 240);
end

% Pick K signals of length L and the size W of the separation window
K = 1;
L = 21;
W = 2*L-1;
X = [ones(ceil(L/2), 1) ; -ones(floor(L/2), 1)];

% Pick a noise level
sigma = 3;

% Desired number of occurrences of each signal X(:, k)
Len_m = 9;
m_want_vector = round(logspace(3,7,Len_m));
error = zeros(Len_m,1);
gamma2 = zeros(Len_m,1);

for qq = 1:Len_m
 
m_want =    m_want_vector(qq); 
% Length of micrograph
n = sum(m_want)*W*10;

fprintf('Micrograph length: %g\n\n\n', n);

%% Pick which correlation coefficients to sample

[list2, list3] = moment_selection(L, 'exclude biased');

%% Generate the micrograph

T = tic();
[y_clean, m_actual] = generate_clean_micrograph_1D_heterogeneous(X, W, n, m_want);
y_obs = y_clean + sigma*randn(n, 1);
time_to_generate_micrograph = toc(T);
fprintf('Time to generate micrograph: %.2g [s]\n', time_to_generate_micrograph);
SNR = norm(y_clean, 'fro')/norm(y_obs-y_clean, 'fro');
fprintf('   SNR: %.2g\n', SNR);
fprintf('   m_actual/m_want: ');
fprintf(' %.2g', m_actual./m_want);
fprintf('\n');

%% Collect the moments
T = tic();
% [M1, M2, M3] = moments_from_data_no_debias_1D(y_obs, list2, list3);
batch_size = 1e8;
[M1, M2, M3] = moments_from_data_no_debias_1D_batch(y_obs, list2, list3, batch_size);
time_to_compute_moments = toc(T);
fprintf('   Moment computation: %.4g [s]\n', time_to_compute_moments);

moments.M1 = M1 / n;  %%%% !!!!! We normalize by n here
moments.M2 = M2 / n;
moments.M3 = M3 / n;
moments.list2 = list2;
moments.list3 = list3;

clear y_clean y_obs;
%ID = randi(1000000);
%filename_data = sprintf('data_example_homogeneous_n_%d_%d', n, ID);
%save([filename_data, '.mat']);

%% Optimization

L_optim = 2*L-1;
sigma_est = 0; % irrelevant if biased terms are excluded and if the weights internally do not depend on sigma

[X2(:,qq), gamma2(qq), X1, gamma1, X1_L, cost_X2] = heterogeneous_1D(moments, K, L, L_optim, sigma_est);
error(qq) = norm(X2(:,qq) - X(:))/norm(X(:));

save('error','error');
save('X2','X2');
fprintf('Relative error = %g\n',error);

%%
% best_error = inf;
% best_P = [];
% permutations = perms(1:K);
% for p = 1 : size(permutations, 1)
%     P = permutations(p, :);
%     relative_error = norm(X-X2(:, P), 'fro') / norm(X, 'fro');
%     if relative_error < best_error
%         best_error = relative_error;
%         best_P = P;
%     end
% end
% P = best_P;
% 
% fprintf('==\n');
% fprintf('Relative error subsignals of length L: %g\n', norm(X-X1_L(:, P), 'fro') / norm(X, 'fro'));
% fprintf('Relative error after reoptimization:   %g\n', norm(X-X2(:, P), 'fro') / norm(X, 'fro'));
% fprintf('Individual relative errors: ');
% fprintf('%g / ', sqrt(sum((X-X2(:, P)).^2, 1)) ./ sqrt(sum(X.^2, 1)));
% fprintf('\b\b\n');
% fprintf('==\n');

% TODO: pick best P and apply it below

% fprintf('Estimated densities (before re optimization):\n');
% disp(gamma1(P)' * (L/L_optim));
fprintf('Estimated densities:\n');
disp(gamma2');
fprintf('True densities:\n');
disp(m_actual*L/n);
save('gamma2','gamma2')
%save([filename_data, '.mat']);

end


%% Requires P as defined above
if 0
figure(1);
T = 0:(L-1);
for k = 1 : K
    subplot(1, K, k);
    %handles = plot(T, X1_L(:, P(k)), T, X2(:, P(k)), T, X(:, k));
    handles = plot(T, X2(:, P(k)), T, X(:, k));
    set(handles(2), 'LineWidth', 1);
    ylim([-2.5, 2.5]);
    title(sprintf('%.2g / %.2g', gamma2(P(k)), m_actual(k)*L/n));
    set(gca, 'YTick', [-2, 0, 2]);
    set(gca, 'XTick', [0, 10, 20]);
    set(gca, 'FontSize', 14);
end
set(gcf, 'Color', 'w');
figname1 = [filename_data '_fig1'];
savefig(1, [figname1, '.fig']);
pdf_print_code(1, [figname1 '.pdf'], 14);
end
