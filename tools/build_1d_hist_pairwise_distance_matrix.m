function [C] = build_1d_hist_pairwise_distance_matrix(hist_size)
% lambda is the ratio between the cost to remove/add mass and the maximum cost otherwise

    %% build the pairwise distance matrix
    
    x = [1:hist_size] - 1;
    x = x';
    
    X = x.*x;
    Y = X';
    C = -2 * (x * x') + repmat(X,[1,hist_size]) + repmat(Y,[hist_size,1]);
    C = sqrt(C);
    
    %creation_cost = lambda*max(max(C));
    %C = [C creation_cost*ones(hist_size,1); creation_cost*ones(1,hist_size) 0];    

 end
