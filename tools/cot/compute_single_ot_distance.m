function [ val, grad ] = compute_single_ot_distance( C, a, b, solver, p, opt)
%COMPUTE_SINGLE_OT_DISTANCE Summary of this function goes here
%   Detailed explanation goes here

if p == 2
    Cmat = C.^2;
else
    Cmat = C;
end

if solver == OTSolver.FastEMD
    assert(p == 1);
    [val, grad] = compute_single_ot_distance_fastemd(Cmat, a, b);
elseif solver == OTSolver.Gurobi
    [val, grad] = compute_single_ot_distance_gurobi(Cmat, a, b);
elseif solver == OTSolver.Linprog
    [val, grad] = compute_single_ot_distance_linprog(Cmat, a, b);
elseif solver == OTSolver.Mosek
    [val, grad] = compute_single_ot_distance_mosek(Cmat, a, b);
else %solver == OTSolver.Sinkhorn
    [val, grad] = compute_single_ot_distance_sinkhorn(Cmat, a, b, opt);
end

if p == 1
    grad = 2*val*grad;
    val = val^2;
end

end
