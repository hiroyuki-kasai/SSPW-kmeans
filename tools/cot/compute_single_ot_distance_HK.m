function [ val, grad ] = compute_single_ot_distance_HK( C, a, b, solver, p, opt)
%COMPUTE_SINGLE_OT_DISTANCE Summary of this function goes here
%   Detailed explanation goes here

    if p == 2
        Cmat = C.^2;
    else
        Cmat = C;
    end
    
    if opt.compress

        a_org = a;
        b_org = b;
        a_nonzero_index = find(a > 0);
        b_nonzero_index = find(b > 0);
        a = a(a_nonzero_index);
        b = b(b_nonzero_index);
        Cmat = Cmat(a_nonzero_index, b_nonzero_index);
        %ratio = (length(a_nonzero_index)+length(b_nonzero_index))/(length(a_org)+length(b_org))
    end

    if solver == OTSolver.FastEMD
        assert(p == 1);
        [val, grad] = compute_single_ot_distance_fastemd(Cmat, a, b);
    elseif solver == OTSolver.Gurobi
        %[val, grad] = compute_single_ot_distance_gurobi(Cmat, a, b);
        [val, grad] = compute_single_ot_distance_gurobi_HK(Cmat, a, b);
    elseif solver == OTSolver.Linprog
        %[val, grad] = compute_single_ot_distance_linprog(Cmat, a, b);
        [val, grad] = compute_single_ot_distance_linprog_HK(Cmat, a, b);
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
