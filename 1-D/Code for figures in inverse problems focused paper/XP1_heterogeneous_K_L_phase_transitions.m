
% Feb 6, 2019, NB
% Current code is noiseless regime rather than infinite data regime, in
% that there is no bias term added to the (perfect) mixed moments.

clear all; %#ok<CLALL>
close all;
clc;

%%

% Ls = 1:1:100; % 1 might be trouble
% Ks = 1:10;
% nrepeats = 30;
Ls = 5:5:100; %[5 20:20:100];
Ks = 1:1:10;
nrepeats = 50;

nmetrics = 3;
metric = zeros(nmetrics, length(Ls), length(Ks), nrepeats);
% Metric 1: relative estimation error
% Metric 2: objective value reached
% Metric 3: CPU time

% opts = struct();
% opts.maxiter = 200;
% opts.tolgradnorm = 1e-10;
% opts.tolcost = 1e-18;
    
fid = fopen('XP1_progress.txt', 'a');
origin = tic();
fprintf(fid, 'Starting: %s\r\n\r\n', datestr(now()));

for iter_K = 1 : length(Ks)
    
    K = Ks(iter_K);
        
    fprintf(fid, 'K = %3d, %s\r\nElapsed: %s [s]\r\n', K, datestr(now()), toc(origin));
    
    for iter_L = 1 : length(Ls)
        
        L = Ls(iter_L);
        
        [list2, list3] = moment_selection(L, 'exclude biased');
        
%         if mod(iter_L, 10) == 0
            fprintf(fid, '\tL = %3d, %s\r\n', L, datestr(now()));
%         end
        
        x_true = randn(L, K);
        y = reshape([x_true ; zeros(2*L, K)], 3*L*K, 1); % the perfect micrograph; 1*L might be enough.
        gamma0 = ones(1, K)*(2*L-1)/length(y);
        n_init_optim = 1;
        
        parfor repeat = 1 : nrepeats
        
            % Solve from a new random initial guess.
            t = tic();
            result = micrographMRA_heterogeneous(y, 0, L, K, list2, list3, ...
                                                 [], gamma0, n_init_optim);
            t = toc(t);

            % Evaluate quality of recovery, up to permutations and shifts.
            x_est = result.X2;
            P = best_permutation(x_true, x_est);
            x_est = x_est(:, P);
            relative_error = norm(x_est - x_true, 'fro') / norm(x_true, 'fro');
            metric(:, iter_L, iter_K, repeat) = [relative_error, result.cost_X2, t];
            
        end
        
    end
    
    save XP1.mat;
    
end

fprintf(fid, 'Ending: %s\r\n\r\nElapsed: %s [s]\r\n', datestr(now()), toc(origin));
fclose(fid);

%%
save XP1.mat;

%%
load XP1;

threshold = 1e-16; % threshold in CISS paper is 1e-16

figure(1);
clf;
metric1 = squeeze(metric(1, :, :, :));
metric2 = squeeze(metric(2, :, :, :));
metric3 = squeeze(metric(3, :, :, :));
% subplot(3, 1, 1);
% imagesc(Ls, Ks, log10(median(metric1, 3))');
% xlabel('L');
% ylabel('K');
% title('Median log10 of recovery error');
% set(gca, 'YDir', 'normal');
% colorbar;
% axis equal;
% axis tight;
subplot(3, 1, 1);
% imagesc(Ls, Ks, log10(min(metric2, [], 3))');
imagesc(Ls, Ks, mean(metric2 <= threshold, 3)');
% xlabel('L');
ylabel('K');
% title('Smallest objective value attained in log10');
title('Fraction of initializations leading to optimality');
set(gca, 'YDir', 'normal');
colorbar;
% axis equal;
pbaspect([5, 1, 1]);
axis tight;

subplot(3, 1, 2);
Q = zeros(length(Ls), length(Ks));
for iter_L = 1 : length(Ls)
    for iter_K = 1 : length(Ks)
        q = find(metric2(iter_L, iter_K, :) <= threshold);
        z = max(squeeze(metric1(iter_L, iter_K, q))); % display max or median?
        if isempty(z)
            z = 1; % if no optimum found, return 0: relative error is 1
        end
        Q(iter_L, iter_K) = z;
    end
end
imagesc(Ls, Ks, log10(Q'));
% xlabel('L');
ylabel('K');
title('Largest relative estimation error among computed optima, log_{10} scale');
set(gca, 'YDir', 'normal');
colorbar;
% axis equal;
pbaspect([5, 1, 1]);
axis tight;

subplot(3, 1, 3);
imagesc(Ls, Ks, median(log10(metric3), 3)');
xlabel('L');
ylabel('K');
title('Median computation time in log_{10} scale');
set(gca, 'YDir', 'normal');
colorbar;
% axis equal;
pbaspect([5, 1, 1]);
axis tight;

set(gcf, 'Color', 'w');

subplot(3, 1, 1); hold all; t = 5:100; plot(t, sqrt(t), 'r-', 'LineWidth', 2); hold off;
subplot(3, 1, 2); hold all; t = 5:100; plot(t, sqrt(t), 'r-', 'LineWidth', 2); hold off;
subplot(3, 1, 3); hold all; t = 5:100; plot(t, sqrt(t), 'r-', 'LineWidth', 2); hold off;

% Bound on how high a K we can expect to be able to demix

% Meaning: strictly above those Ks, it is information-theoretically
% impossible to estimate the signals.
bound_on_K = floor((.5*Ls.*(Ls-1) + 1)./Ls);

for sp = 1 : 3
    subplot(3, 1, sp);
    hold all;
    mask = bound_on_K <= max(Ks);
    plot(Ls(mask), bound_on_K(mask), 'r.', 'MarkerSize', 10);
    hold off;
    xlim([2, max(Ls)]);
end


%%
% subplotsqueeze(gcf, 1.15);

%%
savefig('XP1.fig');
pdf_print_code(gcf, 'XP1.pdf');

