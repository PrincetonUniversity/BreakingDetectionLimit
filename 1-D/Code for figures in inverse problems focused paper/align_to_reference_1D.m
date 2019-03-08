function X_aligned = align_to_reference_1D(X, X_ref)

    C = ifft(conj(fft(X_ref(:))) .* fft(X(:)));
    [~, ind] = max(C);
    X_aligned = circshift(X, 1-ind);

end
