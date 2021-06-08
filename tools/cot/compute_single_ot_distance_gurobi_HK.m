function [ val, grad ] = compute_single_ot_distance_gurobi_HK( C, a, b )
%COMPUTE_SINGLE_OT_DISTANCE Summary of this function goes here
%   Detailed explanation goes here

    n = length(a);
    m = length(b);

%     A = sparse(2*n,n^2);
%     for ii=1:n
%         A(ii,(1:n) + (ii-1)*n) = 1;
%     end
%     for ii=1:n
%         A((n+1):(2*n), (1:n) + (ii-1)*n) = speye(n);
%     end
    
    A = sparse(n+m,n*m);
    for ii=1:n
        A(ii,(1:m) + (ii-1)*m) = 1;
    end
    for ii=1:n
        %A((n+1):(n+m), (ii-1)*m+1:ii*m) = speye(m);
        A((n+1):(n+m), (1:m) + (ii-1)*m) = speye(m);
    end

    C = C';
    
    model.A = [A'; ones(1,n), zeros(1,m)];
    model.obj = [a(:); b(:)];
    model.modelsense = 'max';
    model.rhs = [C(:); 1];
    model.sense = [repmat('<', 1, n*m), '='];
    model.lb = -Inf(n+m,1);

    param = [];
    param.OutputFlag = 0;
    str = gurobi(model, param);

    val = str.objval;
    grad = str.x(1:n);

end