function [ val, grad ] = compute_single_ot_distance_sinkhorn( C, a, b, opt )
%COMPUTE_SINGLE_OT_DISTANCE Summary of this function goes here
%   Detailed explanation goes here

lambda = 1e-3; 
if ~isempty(opt)
    if ~isfield(opt, 'sinkhorn_lambda')
        %lambda = 5e1;
        lambda = 1e-3;        
    else
        lambda = opt.sinkhorn_lambda;
    end
end

K = exp(-lambda*C);
[D,L,u,v] = sinkhornTransport(a(:), b(:), K, K.*C, lambda);
val = D; 

alpha = log(u);
alpha(isinf(alpha)) = 0;
grad = -alpha / lambda;
grad = grad - sum(grad) / length(grad);    

end
