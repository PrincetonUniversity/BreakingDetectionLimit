function [F,J] = costgrad_lsqnonlin_projs_trispect_4tensor(x, x_lists, L,...
    M12_quants, M34_quants, m4_micro, m3_micro, m2_micro, m1_micro, lambda, L_cutoff, gamma)

% parse optimization variable:
% gamma = x(end);
% a_lms = x(1:(end-1)/2) + 1i*x((end-1)/2+1:end-1);
a_lms = x(1:end/2) + 1i*x(end/2+1:end);
a_lms = vec_to_cell_vol_coeffs(a_lms, x_lists.L(x_lists.p));
a_vec = vec_cell(a_lms);

% Compute 1st and 2nd moments:
[m2_harms, G_m2] = power_spectrum_from_harmonics(a_vec, M12_quants.j_l, M12_quants.R_0n, M12_quants.w, L, L_cutoff);
dm2 = gamma*m2_harms - m2_micro;

[m1_harms, G_m1] = mean_from_harmonics(a_vec, M12_quants.j_0, x_lists.s(1), L);
dm1 = gamma*m1_harms - m1_micro;

% Compute gradient of trispectrum slice and bispectrum:

[G_m4, m4_harms] = trispectrum_4tensor_grad_from_harmonics_projs(a_vec,...
    M34_quants.psi_curr, M34_quants.psi_curr_k0, M34_quants.curr_freqs,...
    M34_quants.psi_lNs, M34_quants.q_list, M34_quants.D_mats, L,...
    M34_quants.psi_freqs, M34_quants.a_sizes, L_cutoff);

m4_harms = vec_cell(m4_harms);
dm4 = gamma*m4_harms - m4_micro;

[G_m3, m3_harms] = bispectrum_grad_from_harmonics_projs(a_vec, M34_quants.psi_curr, M34_quants.curr_freqs, M34_quants.psi_lNs, M34_quants.q_list, M34_quants.D_mats, L, M34_quants.psi_freqs, M34_quants.a_sizes, L_cutoff);

m3_harms = vec_cell(m3_harms);

dm3 = gamma*m3_harms - m3_micro;
F = [lambda(1)*real(dm4); lambda(2)*real(dm3); lambda(3)*real(dm2); lambda(4)*real(dm1)];
    
sign_factor = (-1).^(x_lists.L(x_lists.n)+x_lists.m(x_lists.n));

J4 = bsxfun(@times, G_m4(x_lists.n,:), sign_factor) + conj(G_m4(x_lists.p, :));
J3 = bsxfun(@times, G_m3(x_lists.n,:), sign_factor) + conj(G_m3(x_lists.p, :));
J2 = 2*(G_m2(x_lists.p,:) + bsxfun(@times, conj(G_m2(x_lists.n, :)), sign_factor));

J = [lambda(1)*J4, lambda(2)*J3, lambda(3)*J2]; % add m2 derivative
J(x_lists.m(x_lists.p) == 0,:) = (1/2)*J(x_lists.m(x_lists.p) == 0, :); % treat m = 0 case
    
J = [J, lambda(4)*G_m1(x_lists.p)]; % add m1 derivative

J = gamma*[real(J); imag(J)].'; % separate real and imaginary parts
   
% J = [J, real([lambda(1)*m4_harms; lambda(2)*m3_harms; lambda(3)*m2_harms; lambda(4)*m1_harms])]; % add gamma derivative
    

