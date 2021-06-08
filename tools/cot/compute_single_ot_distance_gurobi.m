function [ val, grad ] = compute_single_ot_distance_gurobi( C, a, b )
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

model.A = [A'; ones(1,n), zeros(1,n)];
model.obj = [a(:); b(:)];
model.modelsense = 'max';
model.rhs = [C(:); 1];
model.sense = [repmat('<', 1, n^2), '='];
model.lb = -Inf(2*n,1);

param = [];
param.OutputFlag = 0;
str = gurobi(model, param);

val = str.objval;
grad = str.x(1:n);
    
end