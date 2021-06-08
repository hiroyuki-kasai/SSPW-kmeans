function [ val, grad ] = compute_single_ot_distance_mosek( C, a, b )
%COMPUTE_SINGLE_OT_DISTANCE Summary of this function goes here
%   Detailed explanation goes here

n = length(a);

A = sparse(2*n,n^2);
for ii=1:n
    A(ii,(1:n) + (ii-1)*n) = 1;
end
for ii=1:n
    A((n+1):(2*n), (1:n) + (ii-1)*n) = speye(n);
end

param.MSK_IPAR_LOG = 0;
param.MSK_IPAR_LOG_HEAD = 0;
param.MSK_IPAR_NUM_THREADS = 1;
%param.MSK_IPAR_OPTIMIZER = 'MSK_OPTIMIZER_FREE_SIMPLEX';
%[res] = msklpopt(-[a(:); b(:)], [A'; ones(1,n), zeros(1,n); -ones(1,n), zeros(1,n)], [], [C(:); 1; -1], [], [], param);
% for interior point solver
%val = -res.sol.itr.pobjval;
%grad = res.sol.itr.xx(1:n);

prob.c = -[a(:); b(:)];
prob.a = [A'; ones(1,n), zeros(1,n); -ones(1,n), zeros(1,n)];
prob.blc = [];
prob.buc = [C(:); 1; -1];
prob.blx = [];
prob.bux = [];
[r,res] = mosekopt('minimize echo(0)', prob, param);

% for simplex solver
val = -res.sol.bas.pobjval;
grad = res.sol.bas.xx(1:n);

    
end