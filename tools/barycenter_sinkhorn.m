function centroid = barycenter_sinkhorn(A, M, reg, weights, opts)
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
%     by Hiroyuki Kasai on January 9th, 2020.



    % obtain sizes
    [d, n_hists] = size(A);

    % check arguments    
    if isempty(weights)
        weights = ones(1, n_hists) / n_hists;
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
   

    %M = M / median(median(M)); % suggested by G. Peyre
    K = exp(-M/reg);

    cpt = 0;
    err = 1;

    UKv = K' * (A'./ sum(K))';
    u = (geomean(UKv') ./ UKv')';
    while (err > stopThr && cpt < numItermax)
        cpt = cpt + 1;
        UKv = u .* (K' * (A ./ (K'*u)));
        u = (u .* geobar(weights, UKv')) ./ UKv;
       
        if ~mod(cpt, 10)
            err = sum(std(UKv,1,2));
            
            if verbose
                fprintf('\tbarycenter_sinkhorn: %03d | %.15e\n', cpt, err);
            end
        end

    end

    centroid = geobar(weights, UKv');

end


function m = geobar(weights, alldistribT)

    m = (log(alldistribT))' * weights';
    m = exp(m);

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
