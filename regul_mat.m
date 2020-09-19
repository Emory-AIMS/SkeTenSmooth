function L = regul_mat(coeff_list, shape)
%%function to compute the QV matrix with different coefficients;
% shape is the matrix size

sz = size(coeff_list);
lag = sz(2); % the lag of times

subVec = ones(1, shape-lag);
L = diag(subVec, -lag);
for i = 1:lag
    subvec = ones(1,(shape-(lag-i)));
    subvec = subvec * coeff_list(i);
    subDiag = diag(subvec, -(lag-i));
    L = L + subDiag;
end