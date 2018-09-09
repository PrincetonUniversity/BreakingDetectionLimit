clear; 
close all; 
clc; 

% Script to generate the figure for the PNAS paper\
load ('error');
load ('gamma2');
load('X2');
load('color');

% parameters
xaxis = round(logspace(3,7,9));
L = 21;
W = 2*L-1;
n = xaxis*W*10;
gamma = L./W/10;
error_gamma = abs(gamma2 - gamma)/abs(gamma);
L = 21;
W = 2*L-1;
X = [ones(ceil(L/2), 1) ; -ones(floor(L/2), 1)];

%% plotting
Markersize = 8;
LN = 1.5;
sample_vector = [3,6,9];
ylim_min = -2.1;
ylim_max = 2;

figure; 
subplot(141); hold on;
plot(1:L,X,'linewidth',LN,'color',color(1,:)); 
plot(1:L,X2(:,sample_vector(1)),'linewidth',LN,'color',color(2,:));
axis tight square
%xlabel(['M = ',num2str(xaxis(sample_vector(1)))]);
xlabel(['M = 10^4']);
ylim([ylim_min,ylim_max])
subplot(142); hold on;
plot(1:L,X,'linewidth',LN,'color',color(1,:)); 
plot(1:L,X2(:,sample_vector(2)),'linewidth',LN,'color',color(3,:));
axis tight square
ylim([ylim_min,ylim_max])
%xlabel(['M = ',num2str(xaxis(sample_vector(2)))]);
xlabel(['M = 10^{5.5}']);
subplot(143); hold on;
plot(1:L,X,'linewidth',LN,'color',color(1,:)); 
plot(1:L,X2(:,sample_vector(3)),'linewidth',LN,'color',color(4,:));
axis tight square
ylim([ylim_min,ylim_max])
%xlabel(['M = ',num2str(xaxis(sample_vector(3)))]);
xlabel(['M = 10^7']);
subplot(144);  hold on;
loglog(xaxis, error,'Xb','MarkerSize',Markersize,'linewidth',LN);
loglog(xaxis(sample_vector(1)), error(sample_vector(1)),'o','color',color(2,:),'MarkerSize',Markersize,'linewidth',LN);
loglog(xaxis(sample_vector(2)), error(sample_vector(2)),'o','color',color(3,:),'MarkerSize',Markersize,'linewidth',LN);
loglog(xaxis(sample_vector(3)), error(sample_vector(3)),'o','color',color(4,:),'MarkerSize',Markersize,'linewidth',LN);
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


