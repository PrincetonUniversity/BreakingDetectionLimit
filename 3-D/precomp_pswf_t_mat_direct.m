function [Mt, ang_freq] = precomp_pswf_t_mat_direct(L, beta, T)

[psi_Nn, n_list] = PSWF_2D_full_cart([], L, beta, T);

Mt = cat(2, psi_Nn{:})';

ang_freq = [];
maxN = length(n_list)-1;
for N = -maxN:maxN
    for n = 1:n_list(abs(N)+1)
        ang_freq(end+1,1) = N;
    end
end
