% September 2018
% https://github.com/PrincetonUniversity/BreakingDetectionLimit

clear;
close all;
clc;

% arbitrary (but fixed) seed
seed_rng = rng(57489754);

% use multiple workers if available 
if isempty(gcp('nocreate'))
    parpool(72,'IdleTimeout', 240);
end

%% Define the problem

% input image size
L = 50;

W = 2*L-1;
% Load a grayscale image of size LxL, removing the mean and scale between -1 and 1.
X = double(rgb2gray(imread('Einstein5_small.jpg')));
X = X - mean(X(:));
X = X/max(abs(X(:)));
X = imresize(X, [L, L]);
% Zero padded signal
X_zp = [X zeros(L, W-L) ; zeros(W-L, W)];
% Computing the underlying(of the zero padded image) power spectrum
PSX = abs(fft2(X_zp)).^2;

%noise level
sigma = 3; 
%Desired number of occurrences of the image X in each micrograph. 
%Because of the separation condition, it might be impossible to obtain so
%many occurrences. 
m_want = 1000; 
% Number of micrographs in each iterations. More micrographs in each batch implies more
% efficient computations. 
micrograph_batch = 200; 
% Number of batches (= epochs) of micrographs. 
%Total number of micrographs =  micrograph_batch * epochs.
epochs = 1000;
% Each micrograph is a squared image of size NxN
N = 4096; 
% Threshold for the RRR algorithm. In this experiment, we set it to zero
% so all experiments stop after the same number of iterations.
th = 0; 
% Maximum number of iterations for the RRR algorithm
max_iter = 2000; 
% What images to present in the figures. 
%We chose to present:
%[micrograph_batch, micrograph_batch*10, micrograph_batch*100, micrograph_batch*1000]
ind2save = [1,10,100,1000];

%Initialize variables
err_PS = zeros(epochs,1);
err_rrr = zeros(epochs,max_iter);

% Random initialization for the RRR algorithm
Xs = randn(L);
Xs = Xs - mean(Xs(:));
X_init = zeros(W);
X_init(1:L,1:L) = Xs;

% Save all parameters 
save('parameters');

%% Generate the micrographs and collect their moments

param.L = L ;
param.W = W;
param.sigma = sigma;
param.m_want = m_want;
param.micrograph_batch = micrograph_batch;
param.epochs = epochs;
param.N = N;
param.th = th;
param.max_iter = max_iter;

m = zeros(epochs,1);
for iter = 1:epochs
    
    % Generate micrographs
    Y_obs = zeros(N,N,micrograph_batch);
    tic;
    m_eff = zeros(micrograph_batch,1);
    for i = 1:micrograph_batch
        [Y_clean, m_eff(i)] = generate_clean_micrograph_2D(X, param);
        Y_obs(:,:,i) = Y_clean + sigma*randn(N);
    end
    fprintf('iter = %d \n', iter);
    fprintf('time to generate data = %.4g [sec] \n', toc);
    m(iter) = sum(m_eff);
    tic
    
    % Compute the power spectrum (second-order moment) of a batch of
    % micrographs
    M2_micrograph = computeM2(Y_obs,m_eff,param);
    fprintf('time to compute moments = %.4g [sec] \n', toc);

    if iter == 1
        M2 =  M2_micrograph;
    else
        M2 = M2 + M2_micrograph;
    end
    PS = fft2(M2/iter);
    
    % Relative error of power spectrum estimation
    err_PS(iter) = norm(PSX(:) - PS(:))/norm(PSX(:));
    fprintf('error PS = %.4g\n',err_PS(iter));
    tic
    
    % Run the RRR algorithm
    [Xest_rrr, discrepancy_norm,err,err1, err2] = RRR(sqrt(PS),X_init,X,param);
    fprintf('RRR time = %.4g [sec] \n', toc);
    err_rrr(iter,:) = err;
    fprintf('error RRR = %.4g\n',err_rrr(iter,end));
    
    % Save data of current iteration
    save(strcat('Xest_',num2str(iter),'.mat'),'Xest_rrr');
    save('err_PS','err_PS');
    save('err_rrr','err_rrr');
    
end


%% Plotting the progress figure

% Save the figure as a pdf
save_pdf = 0;
% Load saved data
load_data = 0;
if load_data
    load('parameters.mat')
    load('err_PS');
    load('err_rrr');
end

figure(10);
for i = 1:length(ind2save)
    
    str = strcat('Xest_',num2str(ind2save(i)),'.mat');
    load(str);
    Xest_rrr = Xest_rrr(1:size(X,1),1:size(X,2));
    err1 = norm(Xest_rrr - X,'fro')/norm(X(:));
    err2 = norm(rot90(Xest_rrr,2) - X,'fro')/norm(X(:));
    
    % Correcting the reflection symmetry (if needed)
    if err2<err1 
        Xest_rrr = rot90(Xest_rrr,2);
    end
    subplot(1,length(ind2save),i); 
    imagesc(Xest_rrr); 
    colormap gray; 
    axis tight square off
    
end

if save_pdf
    str = strcat('Einstien_progress_examples');
    pdf_print_code(gcf, str, 12)
end

%% Error curve figures

LN = 1.5;
figure(11);
subplot(121); loglog((1:epochs)*micrograph_batch,err_PS,'b','linewidth',LN);
ylabel('relative error of the autocorrelation');
xlabel('# micrographs')
xlim([micrograph_batch,micrograph_batch*epochs])
ylim([10^(-3),5*10^(-2)])
xticks([10^2,10^3,10^4,10^5])
xticklabels({'10^2','10^3','10^4','10^5'})
axis square
grid on

subplot(122); loglog((1:epochs)*micrograph_batch,err_rrr(1:epochs,end)...
    ,'b','linewidth',LN);
ylabel('relative error of the signal');
xlabel('# micrographs')
xlim([micrograph_batch,micrograph_batch*epochs])
ylim([.15,.7])
axis square
xticks([10^2,10^3,10^4,10^5])
xticklabels({'10^2','10^3','10^4','10^5'})
grid on

if save_pdf
    pdf_print_code(gcf, 'Einstein_recovery_error_combined', 18)
end

