function [R_Nn, alpha_Nn] = PSWF_radial_2D(N, n, c, radial_eval_pts)
% Generate radial 2D PSWF's and their associated eigenvalues
% 
% Inputs:
%   * N: angular frequency
%   * n: maximum radial frequency
%   * c: bandlimit
%   * radial_eval_pts: radial points for evaluation of the radial PSWFs
% 
% Outputs:
%   * R_Nn: maxtrix of size length(radial_eval_pts) x (n + 1), each column
%   representing radial PSWF of radial frequency (column # - 1) evaluated
%   on radial_eval_pts
%   * alpha_Nn: vector of size (n + 1) with the eigenvalues associated to
%   the 2D PSWF of radial frequency (column # - 1).
% 
% Boris Landa, written for:
% 
% Landa, B., & Shkolnisky, Y. (2017). Approximation scheme for essentially bandlimited and space-concentrated functions on a disk. 
% Applied and Computational Harmonic Analysis, 43(3), 381-403.
% 
% Landa, B., & Shkolnisky, Y. (2017). Steerable principal components for space-frequency localized images. SIAM journal on imaging sciences, 10(2), 508-534.
% 

%% Generate points for numerical integration
% persistent x;
% persistent w;
phi_approx_err = 1e-16;
% if (isempty(x))
    [x,w]=lgwt(20*n,0,1);    % Legendre-Gauss nodes for numerical integration   
    x = flipud(x);
    w = flipud(w);    
% end

%% Calculate approx_len
% - Definition for the approximation function of the d's decay
d_decay_approx_fun = @(N,n,c,j) c^2./(16*(j.^2 + j*(2*n+N+1)) - c^2);

% - Finding the first point of the decay according to the approximation
first_idx_for_decrease = ceil((((2*n+N+1)^2+c^2/2)^0.5 - (2*n+N+1))/2);     % Analytic solution for d_decay_approx_fun<1 for every j>first_idx_for_decrease

% - Finding the number of points needed to achieve the chosen error
d_approx = d_decay_approx_fun(N,n,c,first_idx_for_decrease);
d_decay_idx_counter = first_idx_for_decrease;

while(d_approx>phi_approx_err)
    d_decay_idx_counter = d_decay_idx_counter + 1;
    d_approx = d_approx * d_decay_approx_fun(N,n,c,d_decay_idx_counter);
end

approx_len = n + 1 + d_decay_idx_counter;

%% Funcion defenitions
Pn = @(n,alpha,beta,x) j_polynomial(length(x),n,alpha,beta,x);

%% Generate Matrix for calculation of d_k coefficients
B_N = Generate_BN_mat(N,c,approx_len);
[d_vec, xi] = eig(B_N);

%% Sort eigenvalues by descending order
[xiVec,idx] = sort(diag(xi),'descend');
d_vec = d_vec(:,idx);
for i=1:approx_len
    xi(i,i) = xiVec(i);
end

%% Compare true decay of d's with approximation function
% d_decay_compare = [abs(d_vec((n+2):end,n+1)./d_vec((n+1):(end-1),n+1)),d_decay_approx_fun(N,n,c,0:(approx_len-2-n)).'];
% d_approx = cumprod([ones(first_idx_for_decrease,1);d_decay_approx_fun(N,n,c,first_idx_for_decrease:(approx_len-1-n)).']);
% d_decay_compare = [abs(d_vec((n+1):end,n+1)),d_approx];
% plot(d_decay_compare); grid on; title(['N=',num2str(N),', n=',num2str(n),', c=',num2str(c)] );

%% Generate T_n basis functions on grid points ,and the derivatives of T_n.
j=0:(approx_len-1);

% - Calculation of functions
T_x_mat = @(x,N,j,approx_len) ((x.^(N+1/2))*((2*(2*j+N+1)).^(1/2))) .* Pn(approx_len-1,N,0,1-2*x.^2);
% - Calculation of derivatives
T_x_derivative_mat = ((x.^(N+3/2))*(-2*(N+j+1).*((2*(2*j+N+1)).^(1/2)))).*[zeros(length(x),1),Pn(approx_len-2,N+1,1,1-2*x.^2)] + (N+1/2)*((x.^(N-1/2))*((2*(2*j+N+1)).^(1/2))).*Pn(approx_len-1,N,0,1-2*x.^2);

%% Calculate phi - Generalized PSWF's ,and the derivatives of phi.
phi = T_x_mat(x,N,j,approx_len) * d_vec(:,1:(n+1));
% phi = phi .* (ones(length(x),1)*sign(phi(300,:)));    %Fix the signs so that all functions begin positive.

phi_derivatives = T_x_derivative_mat * d_vec(:,1:(n+1));
% phi_derivatives = phi_derivatives .* (ones(length(x),1)*sign(phi(300,:)));    %Fix the signs of the derivatives to be consistant with the signs of the functions.

%% Calculate original problem eigenvalues gamma_N,n

% - First (largest) eigenvalue numerical calculation
[max_phi_val,max_phi_x_idx] = max(abs(phi(:,1)));
x_for_calc = x(max_phi_x_idx);

K_operator = @(nu,x) besselj(nu,x).*sqrt(x);
Right_hand_side_integral = (w.*K_operator(N,c*x_for_calc*x)).' * phi(:,1);
lambda_N_1 = Right_hand_side_integral/phi(max_phi_x_idx,1);

% - Calculate the rest of the eigenvalues based on the first
upper_integral_values = diag( (((w.*x)*ones(1,n)).*phi_derivatives(:,1:(end-1))).' * phi(:,2:end) );
lower_integral_values = diag( (((w.*x)*ones(1,n)).*phi(:,1:(end-1))).' * phi_derivatives(:,2:end) );
lambda_N = [lambda_N_1 ; lambda_N_1 * cumprod(upper_integral_values./lower_integral_values)];

alpha_Nn = lambda_N*2*pi*((1i)^N) / sqrt(c);

%% Test orthonormality of phi's
% Inner_prod_matrix = ((w*ones(1,n+1)).*phi).' * phi;
% bar(abs(eig(Inner_prod_matrix)));

%% Compute PSFW's on the grid inside the disk
j=0:(approx_len-1);
T_radial_part_mat = @(x,N,j,approx_len) ((x.^N)*((2*(2*j+N+1)).^(1/2))) .* Pn(approx_len-1,N,0,1-2*x.^2); % - These are the ordinairy T's divided by x^(1/2).
d_vec = bsxfun(@times,d_vec,sign(sign(diag(d_vec).')+0.5));    %Fix the signs so that all functions begin positive.
R_Nn = T_radial_part_mat(radial_eval_pts,N,j,approx_len) * d_vec(:,1:(n+1));
R_Nn = bsxfun(@times, R_Nn, (1/2)*alpha_Nn.');


end