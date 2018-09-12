% September 2018
% https://github.com/PrincetonUniversity/BreakingDetectionLimit
%
% This script generates figures for the example.m script

clear; 
close all; 
clc; 

%% load variables 

load ('error');
load ('gamma');
load('Xest');
load('color');
load('parameters');
n = m_want_vector*W*10;

%% error of gamma estimation

gamma_true = L./W/10;
error_gamma = abs(gamma - gamma_true)/abs(gamma_true);

%% plotting

Markersize = 8;
LN = 1.5;
sample_vector = [3,6,9];
ylim_min = -2.1;
ylim_max = 2;

figure; 
subplot(141); hold on;
plot(1:L,X,'linewidth',LN,'color',color(1,:)); 
plot(1:L,Xest(:,sample_vector(1)),'linewidth',LN,'color',color(2,:));
axis tight square
xlabel(['M = 10^4']);
ylim([ylim_min,ylim_max])
subplot(142); hold on;
plot(1:L,X,'linewidth',LN,'color',color(1,:)); 
plot(1:L,Xest(:,sample_vector(2)),'linewidth',LN,'color',color(3,:));
axis tight square
ylim([ylim_min,ylim_max])
xlabel(['M = 10^{5.5}']);
subplot(143); hold on;
plot(1:L,X,'linewidth',LN,'color',color(1,:)); 
plot(1:L,Xest(:,sample_vector(3)),'linewidth',LN,'color',color(4,:));
axis tight square
ylim([ylim_min,ylim_max])
xlabel(['M = 10^7']);
subplot(144);  hold on;
loglog(m_want_vector, error,'Xb','MarkerSize',Markersize,'linewidth',LN);
loglog(m_want_vector(sample_vector(1)), error(sample_vector(1)),'o','color',color(2,:),'MarkerSize',Markersize,'linewidth',LN);
loglog(m_want_vector(sample_vector(2)), error(sample_vector(2)),'o','color',color(3,:),'MarkerSize',Markersize,'linewidth',LN);
loglog(m_want_vector(sample_vector(3)), error(sample_vector(3)),'o','color',color(4,:),'MarkerSize',Markersize,'linewidth',LN);
set(gca, 'XScale', 'log')
set(gca, 'YScale', 'log')
axis tight square
xlabel('M')
ylabel('relative error')
xticks([10^3, 10^4, 10^5, 10^6, 10^7])
xticklabels({'10^3', '10^4', '10^5', '10^6', '10^7'})

%pdf_print_code(gcf, '1D_example.pdf', 12)

fprintf('gamma error = %g,%g,%g\n',error_gamma(sample_vector(1))...
    ,error_gamma(sample_vector(2)),error_gamma(sample_vector(3)));


