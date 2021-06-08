function [ val, grad ] = compute_single_ot_distance_fastemd( C, a, b )
%COMPUTE_SINGLE_OT_DISTANCE Summary of this function goes here
%   Detailed explanation goes here

n = length(a);

% [val, F] = emd_hat_gd_metric_mex(a(:),b(:),C,-1);%,FType)
% uv = dual_variables(C, F);
% grad = uv(1:n) - 1/n*sum(uv(1:n));

% assume C is already made up of _squared_ distances
[val_unsq, F] = emd_hat_mex(a(:),b(:),sqrt(C),-1);%,FType)
%[val_unsq, F] = emd_hat_gd_metric_mex(a(:),b(:),sqrt(C),-1);%,FType)
uv = dual_variables(C, F);
grad_unsq = uv(1:n) - 1/n*sum(uv(1:n));

val = val_unsq;%^2;
grad = grad_unsq;%2*val_unsq * grad_unsq;
end

function [uv] = dual_variables(cost, flow)
n1 = size(flow,1);
n2 = size(flow,2);

[I,J,~] = find(flow);
A = sparse(length(I)+1, n1+n2);
for kk=1:length(I)
    A(kk, I(kk)) = 1;
    A(kk, n1 + J(kk)) = 1;
end
A(length(I)+1,1) = 1;
C_sub = zeros(length(I)+1,1);
for kk=1:length(I)
    C_sub(kk) = cost(I(kk),J(kk));
end
warning('off');
uv = A\C_sub;
warning('on');
end
