function [beta_star, S_star] = GSSP(w, lambda, k)
% =========================================================================
%                   Sparse projection onto the positive simplex
% =========================================================================
% INPUT ARGUMENTS:
% w                         n x 1 anchor point
% lambda                    lambda = 1: probability simplex.
% k                         Sparsity                                
% =========================================================================
% OUTPUT ARGUMENTS:
% beta_star                 k-sparse solution satisfying simplex constraint
% S_star                    Support of solution
% =========================================================================
% 01/04/2012, by Anastasios Kyrillidis. anastasios@utexas.edu
% =========================================================================

[~, ind] = sort(w, 'descend');
S_star = ind(1:k);

beta_star = zeros(size(w));

tau = (1/k)*(lambda - sum(w(S_star)));
beta_star(S_star) = max(w(S_star) + tau.*ones(size(w(S_star))), 0);