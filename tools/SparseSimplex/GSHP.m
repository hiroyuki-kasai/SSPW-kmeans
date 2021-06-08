function [beta_star, S_star] = GSHP(w, lambda, k)
% =========================================================================
%                   Sparse projection onto the simplex
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
%% First step
[~, i] = max(sign(lambda).*w);
S = i;

%% Second step
for l = 2:k
    sum_wS = sum(w(S));
    obj_vec = abs(w - (sum_wS - lambda)/(l - 1));    
    [~, I] = sort(obj_vec, 'descend');    
    i = 1;
    while (ismember(I(i), S))
        i = i + 1;
    end;    
    S = [S I(i)];
end;

S_star = S;
beta_star = zeros(size(w));
tau = (1/k)*(lambda - sum(w(S_star)));
beta_star(S_star) = w(S_star) + tau.*ones(size(w(S_star)));