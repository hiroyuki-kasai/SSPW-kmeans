function [ val, grad ] = compute_single_ot_distance_linprog( C, a, b )
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

options = optimset('linprog');
options.Display = 'off';
problem.options = options;
problem.solver = 'linprog';
problem.f = -[a(:); b(:)];
problem.Aineq = A';
problem.bineq = C(:);
problem.Aeq = [ones(1,n), zeros(1,n)];
problem.beq = 1;
%[x,fval] = linprog(problem);
[x,fval] = linprog(-[a(:); b(:)], A', C(:), [ones(1,n), zeros(1,n)], 1);
val = -fval;
grad = x(1:n);
    
end