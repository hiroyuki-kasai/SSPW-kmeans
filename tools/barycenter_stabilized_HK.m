function centroid = barycenter_stabilized_HK(A, M, reg, tau, weights, numItermax, stopThr)
%The function solves the following optimization problem:
%
%    .. math::
%       \mathbf{a} = arg\min_\mathbf{a} \sum_i
%       W_{reg}(\mathbf{a},\mathbf{a}_i)%
%
%    where :
%
%    - :math:`W_{reg}(\cdot,\cdot)` is the entropic regularized Wasserstein distance (see ot.bregman.sinkhorn)
%    - :math:`\mathbf{a}_i` are training distributions in the columns of matrix :math:`\mathbf{A}`
%    - reg and :math:`\mathbf{M}` are respectively the regularization term
%    and the cost matrix for OT%
%
%    The algorithm used for solving the problem is the Sinkhorn-Knopp matrix scaling algorithm as proposed in [3]_
%
%    Parameters
%    ----------
%    A : ndarray, shape (dim, n_hists)
%        n_hists training distributions a_i of size dim
%    M : ndarray, shape (dim, dim)
%        loss matrix for OT
%    reg : float
%        Regularization term > 0
%    tau : float
%        thershold for max value in u or v for log scaling
%    weights : ndarray, shape (n_hists,)
%        Weights of each histogram a_i on the simplex (barycentric coodinates)
%    numItermax : int, optional
%        Max number of iterations
%    stopThr : float, optional
%        Stop threshol on error (>0)
%    verbose : bool, optional
%        Print information along iterations
%    log : bool, optional
%        record log if True
%
%
%    Returns
%    -------
%    a : (dim,) ndarray
%        Wasserstein barycenter
%    log : dict
%        log dictionary return only if log==True in parameters
%
%
%   References
%  ----------
%
%    .. [3] Benamou, J. D., Carlier, G., Cuturi, M., Nenna, L., & Peyr?, G. (2015). Iterative Bregman projections for regularized transportation problems. SIAM Journal on Scientific Computing, 37(2), A1111-A1138.
%
%   """
    verbose  = true;
    numItermax = 1000;
    stopThr = 1e-4;
    %tau = 1e10;
    [dim,n_hists] = size(A);
    
    if isempty(weights)
        weights = ones(1,n_hists) / n_hists;
    end

    tmp = -M/reg;
    K = exp(-M/reg);

    %u = ones(dim, n_hists)/dim;
    v = ones(dim, n_hists)/dim;

    cpt = 0;
    err = 1.;
    alpha = zeros(dim , 1);
    beta = zeros(dim, 1);
    q = ones(dim, 1)/dim;

    while (err > stopThr && cpt < numItermax)
        qprev = q;
        Kv = K * v;
        u = A ./ (Kv+1e-16);
        Ktu = K' * u;
        q = geobar(weights,Ktu');
        
        if mod(cpt,10) == 0
            %disp(cpt);
            %X = sprintf('%.15e\n',q);
            %disp(X);
        end
        
        Q = repmat(q,1,n_hists);
        v = Q ./(Ktu+1e-16);
        absorbing = 0;
        if  any(u(:) > tau) || any(v(:) > tau)
            absorbing = 1;
            alpha = alpha +reg * log(max(u,[],2));
            beta = beta + reg * log(max(v,[],2));
            Alpha = repmat(alpha,1,dim);
            Beta = repmat(beta,1,dim);
            K = exp(( Alpha + Beta - M) / reg);
            sz = size(v);
            v = ones(sz, 'like',v);
        end
        
        Kv = K * v;
        
        if  any(Ktu(:) == 0.)||any(isnan(u(:)))||any(isnan(v(:)))||any(isinf(u(:)))||any(isinf(v(:)))
            q = qprev;
            break;
        end
        
        if (mod(cpt,1)==0 && ~(absorbing))||(cpt == 0)
            media = abs(u .* Kv - A);
            err = max(media(:));
            if verbose
                fprintf('%03d | %.15e\n', cpt, err);
            end
        end
        cpt = cpt + 1;
    end

    centroid = q;

end

function m = geobar(weights, alldistribT)

    m = (log(alldistribT))' * weights';
    m = exp(m);

    %return np.exp(np.dot(np.log(alldistribT), weights.T))
end

function m = geomean(x,dim)
%GEOMEAN Geometric mean.
%   M = GEOMEAN(X) returns the geometric mean of the values in X.  When X
%   is an n element vector, M is the n-th root of the product of the n
%   elements in X.  For a matrix input, M is a row vector containing the
%   geometric mean of each column of X.  For N-D arrays, GEOMEAN operates
%   along the first non-singleton dimension.
%
%   GEOMEAN(X,'all') is the geometric mean of all the elements of X.
%
%   GEOMEAN(X,DIM) takes the geometric mean along dimension DIM of X.
%
%   GEOMEAN(X,VECDIM) finds the geometric mean of the elements of X based 
%   on the dimensions specified in the vector VECDIM.
%
%   See also MEAN, HARMMEAN, TRIMMEAN.

%   Copyright 1993-2018 The MathWorks, Inc.


    if any(x(:) < 0) || ~isreal(x)
        error(message('stats:geomean:BadData'))
    end

    if nargin < 2 || isempty(dim)
        % Figure out which dimension sum will work along.
        dim = find(size(x) ~= 1, 1);
        if isempty(dim), dim = 1; end
    end

    % Validate dim
    if isnumeric(dim)
        if ~isreal(dim) || any(floor(dim) ~= ceil(dim)) || any(dim < 1) || any(~isfinite(dim))
            error(message('MATLAB:getdimarg:invalidDim'));
        end
        if ~isscalar(dim) && ~all(diff(sort(dim)))
            error(message('MATLAB:getdimarg:vecDimsMustBeUniquePositiveIntegers'));
        end
    elseif ((ischar(dim) && isrow(dim)) || ...
     (isstring(dim) && isscalar(dim) && (strlength(dim) > 0))) && ...
     strncmpi(dim,'all',max(strlength(dim), 1))
            dim = 'all';
    else
        error(message('MATLAB:getdimarg:invalidDim'));
    end

    n = getSize(x,dim);
    % Prevent divideByZero warnings for empties, but still return a NaN result.
    if n == 0, n = NaN; end

    % Take the n-th root of the product of elements of X, along dimension DIM.
    if nargin < 2
        m = exp(sum(log(x))./n);
    else
        m = exp(sum(log(x),dim)./n);
    end

end

function s = getSize(x, dim)
    if isnumeric(dim)
        s = size(x);
        dim = dim(dim <= numel(s));
        s = prod(s(dim));
    else
        s = numel(x);
    end

end



































