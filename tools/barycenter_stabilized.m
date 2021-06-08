function centroid = barycenter_stabilized(A, M, reg, tau, weights, opts)
%     The function solves the following optimization problem:
% 
%     .. math::
%        \mathbf{a} = arg\min_\mathbf{a} \sum_i W_{reg}(\mathbf{a},\mathbf{a}_i)
% 
%     where :
% 
%     - :math:`W_{reg}(\cdot,\cdot)` is the entropic regularized Wasserstein distance (see ot.bregman.sinkhorn)
%     - :math:`\mathbf{a}_i` are training distributions in the columns of matrix :math:`\mathbf{A}`
%     - reg and :math:`\mathbf{M}` are respectively the regularization term and the cost matrix for OT
% 
%     The algorithm used for solving the problem is the Sinkhorn-Knopp matrix scaling algorithm as proposed in [3]_
% 
%     Parameters
%     ----------
%     A : hisotogram matrix of size (d x n_hists)
%         n_hists training distributions a_i of size d
%     M : loss matrix for OT of size (d x d)
%     reg : Regularization term > 0
%     tau : Thershold for max value in u or v for log scaling
%     weights : vector of size n_hists
%         Weights of each histogram a_i on the simplex (barycentric coodinates)
%     numItermax : int, optional
%         Max number of iterations
%     stopThr : Stop threshol on error (>0)
%     verbose : 
% 

%     Returns
%     -------
%     centroid : Wasserstein barycenter
% 
% 
%     References
%     ----------
% 
%     .. [3] Benamou, J. D., Carlier, G., Cuturi, M., Nenna, L., & Peyr?, G. (2015). Iterative Bregman projections for regularized transportation problems. SIAM Journal on Scientific Computing, 37(2), A1111-A1138.
% 
%     
%     This code is ported from bregman.py in POT (https://github.com/rflamary/POT) 
%     by Takumi Fukunaga and Hiroyuki Kasai on January 9th, 2020.

    
    % obtain sizes
    [d, n_hists] = size(A);

    % check arguments    
    if isempty(weights)
        weights = ones(1,n_hists) / n_hists;
    end
    
    if ~isfield(opts, 'numItermax')
        numItermax = 1000;
    else
        numItermax = opts.numItermax;
    end    
    
    if ~isfield(opts, 'stopThr')
        stopThr = 1e-4;
    else
        stopThr = opts.stopThr;
    end  
    
    if ~isfield(opts, 'verbose')
        verbose = false;
    else
        verbose = opts.verbose;
    end      
   

    K = exp(-M/reg);

    %u = ones(d, n_hists)/d;
    v = ones(d, n_hists)/d;
    alpha = zeros(d , 1);
    beta = zeros(d, 1);
    q = ones(d, 1)/d;
    
    cpt = 0;
    err = 1.;
    while (err > stopThr && cpt < numItermax)
        qprev = q;
        Kv = K * v;
        u = A ./ (Kv + 1e-16);
        Ktu = K' * u;
        q = geobar(weights, Ktu');
        
        Q = repmat(q, 1, n_hists);
        v = Q ./(Ktu + 1e-16);
        
        absorbing = 0;
        if  any(u(:) > tau) || any(v(:) > tau)
            absorbing = 1;
            alpha = alpha +reg * log(max(u,[],2));
            beta = beta + reg * log(max(v,[],2));
            Alpha = repmat(alpha,1,d);
            Beta = repmat(beta,1,d);
            K = exp(( Alpha + Beta - M) / reg);
            sz = size(v);
            v = ones(sz, 'like', v);
        end
        
        Kv = K * v;
        
        if  any(Ktu(:) == 0.) || any(isnan(u(:))) || any(isnan(v(:))) || any(isinf(u(:)))||any(isinf(v(:)))
            q = qprev;
            break;
        end
        
        if (mod(cpt, 10)==0 && ~(absorbing))||(cpt == 0)
            media = abs(u .* Kv - A);
            err = max(media(:));
            
            if verbose
                fprintf('\tbarycenter_stabilized: %03d | %.15e\n', cpt, err);
            end
        end
        
        cpt = cpt + 1;
    end

    centroid = q;

end

function m = geobar(weights, alldistribT)

    m = (log(alldistribT))' * weights';
    m = exp(m);

end



































