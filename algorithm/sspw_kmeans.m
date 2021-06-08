function [centroids, info, init_centroids] = sspw_kmeans(C, X, init_centroids, init_labels, opts)
% Wasserstein k-means algorithm.
%
% Inputs:
%       C                   cost matrix
%       X                   samples
%       init_centroids      initial centroids
%       init_labels         initial labels
%       opts                options
% Output:
%       centroids           centroids
%       info                statistics
%       init_centroids      initial centroids used
%
% References:
%
%
% Note: 
%       Some parts in find_closest_centroid originally comes from 
%       https://github.com/mstaib/cloud-regime-clustering-code
%
%
% Created by H.Kasai on Dec. 26, 2019
% Modified by H.Kasai on Mar. 04, 2020



    proj_simplex_helper = proj_simplex(); %TFOCS
    function [x_out] = proj(x)
        [~, x_out] = proj_simplex_helper(x, 1);
    end

    oracle = @(x,y) compute_single_ot_distance_HK(C, x, y, opts.OTSolver, opts.p, opts);
    
    
    %% check opts
    if ~isfield(opts, 'barycenter_lambda')
        opts.barycenter_lambda = 0.005;
    else
        %
    end 
    
    if ~isfield(opts, 'barycenter_alg')
        opts.barycenter_alg = 'sinkhorn';
    else
        %
    end     
    

    %% initialize centroids
    if isempty(init_centroids)
        if strcmp(opts.init_alg, 'kmeans')
            %[init_labels, init_centroids] = litekmeans(X, opts.k, 'Replicates', 1, 'clusterMaxIter', 5);
            [init_labels, init_centroids] = litekmeans(X, opts.k, 'Replicates', 20);
        else
            init_centroids = X(randsample(size(X,1), opts.k), :);
            init_labels = [];
        end
    else
        
    end
    centroids = init_centroids;
    labels = init_labels;
    barycenter_opts = [];
    
    
    % Update sparse_ratio
    if opts.sparse_centroid || opts.sparse_sample
        if strcmp(opts.sparse_ratio_alg, 'fix')
            sparse_ratio = opts.sparse_ratio;
        elseif strcmp(opts.sparse_ratio_alg, 'dec')
            sparse_ratio = 1.0;
        elseif strcmp(opts.sparse_ratio_alg, 'inc')
            sparse_ratio = opts.sparse_ratio;
        else
            sparse_ratio = opts.sparse_ratio;
        end
    else
        sparse_ratio = 1.0;
    end


    % Project centroids onto simplex
    for j = 1 : opts.k
        if opts.sparse_centroid        
            original_cardinality = nnz(centroids(j, :));
            opts.sparse_cardinality = round(original_cardinality * sparse_ratio);
            if strcmp(opts.sparse_alg, 'GSHP')
                [centroids(j, :), S_star] = GSHP(centroids(j, :), 1, opts.sparse_cardinality);
            else
                [centroids(j, :), S_star] = GSSP(centroids(j, :), 1, opts.sparse_cardinality);
            end
        else
            centroids(j, :) = proj(centroids(j, :));
        end   
    end

    if ~isempty(labels)
        accuracy = eval_clustering_accuracy(opts.gnd, labels);
        fprintf('# W-kmeans: Ini, purity=%.4f, nmi=%.4f, acc=%.4f, src=%.2f\n', accuracy.purity, accuracy.nmi, accuracy.acc, sparse_ratio);
    end

    % reset labels
    labels = zeros(size(X,1), 1);
    
    need_find_labels = true;
    total_label_update_time = 0;
    total_centroid_update_time = 0;    
    
    if opts.verbose > 2
        start_time = tic();
        [labels, distance, ~] = find_closest_centroid(centroids, oracle, X, sparse_ratio, opts);
        label_update_time = toc(start_time); 
        total_label_update_time = total_label_update_time + label_update_time;
        accuracy = eval_clustering_accuracy(opts.gnd, labels);
        fprintf('# W-kmeans: 000, dist=%.8f, purity=%.4f, nmi=%.4f, acc=%.4f, label_t=%.1f, cent_t=na\n', ...
            sum(distance), accuracy.purity, accuracy.nmi, accuracy.acc, label_update_time);
        need_find_labels = false;
    end 
    
    % set initial distance
    prev_distance = inf;

    %% main loop
    for iter = 1 : opts.max_iter
        
        start_time = tic();
        
        % Update sparse_ratio
        if opts.sparse_centroid || opts.sparse_sample
            if strcmp(opts.sparse_ratio_alg, 'fix')
                sparse_ratio = opts.sparse_ratio;
            elseif strcmp(opts.sparse_ratio_alg, 'dec')
                sparse_ratio = 1.0 - (1.0 - opts.sparse_ratio)/opts.max_iter * iter;
            elseif strcmp(opts.sparse_ratio_alg, 'inc')
                sparse_ratio = opts.sparse_ratio + (1.0 - opts.sparse_ratio)/opts.max_iter * iter;
            else
                sparse_ratio = opts.sparse_ratio;
            end
        end
        
        % Update labels
        prev_labels = labels;
        if need_find_labels
            [labels, distance, ~] = find_closest_centroid(centroids, oracle, X, sparse_ratio, opts);         
        else
            need_find_labels = true;
            prev_distance = inf;
        end
        
        thresh_val = 0.99 * sum(distance);
        if prev_distance <= sum(distance)
            labels = prev_labels;
            fprintf('# W-kmeans: stopping due to increase of distance (%.4f < %.4f).\n', prev_distance, sum(distance));
            break;
        else
        
            label_update_time = toc(start_time); 
            total_label_update_time = total_label_update_time + label_update_time;        

            % Update centroids
            start_time = tic();
            for j = 1 : opts.k
                class_index = find(labels == j);
                if ~isempty(class_index)
                    samples = X(class_index, :);
                    if strcmp(opts.barycenter_alg, 'sinkhorn')
                        tmp = barycenter_sinkhorn(samples', C, opts.barycenter_lambda, [], barycenter_opts);
                    else
                        tmp = barycenter_stabilized(samples', C, opts.barycenter_lambda, 1e10, [], barycenter_opts);
                    end
                    tmp(isinf(tmp)) = 0;
                    centroids(j, :) = tmp';

                    if opts.sparse_centroid
                        original_cardinality = nnz(centroids(j, :));
                        sparse_cardinality = round(original_cardinality * sparse_ratio);

                        if strcmp(opts.sparse_alg, 'GSHP')
                            [centroids(j, :), ~] = GSHP(centroids(j, :), 1, sparse_cardinality);
                        elseif strcmp(opts.sparse_alg, 'GSSP') 
                            [centroids(j, :), ~] = GSSP(centroids(j, :), 1, sparse_cardinality);
                        elseif strcmp(opts.sparse_alg, 'Simp') 
                            [centroids(j, :), ~] = simple_simplex_sparse(centroids(j, :), sparse_cardinality);             
                        end

                    else
                        %tmp = centroids(j, :);
                        %centroids(j, :) = tmp;
                        % do nothing
                    end
                end

                TF = isnan(centroids(j, :));
                if nnz(TF)
                    fprintf('# W-kmeans: error: centroids has NAN in wasserstein_kmeans function.\n');
                    return;
                end      

                minus_set = find(centroids(j, :) < 0, 1);
                if ~isempty(minus_set)
                    fprintf('# W-kmeans: error: centroids has negative values in wasserstein_kmeans function.\n');
                    return;
                end            
            end
            centroid_update_time = toc(start_time);
            total_centroid_update_time = total_centroid_update_time + centroid_update_time;
            prev_distance = sum(distance);

            if opts.verbose > 1
                if opts.verbose > 2
                    accuracy = eval_clustering_accuracy(opts.gnd, labels);
                    fprintf('# W-kmeans: %03d, dist=%.8f, purity=%.4f, nmi=%.4f, acc=%.4f, label_t=%.1f, cent_t=%.1f, scr=%.2f\n', ...
                        iter, sum(distance), accuracy.purity, accuracy.nmi, accuracy.acc, label_update_time, centroid_update_time, sparse_ratio);
                end 

            end
        end
        
    end

    info.labels = labels;
    info.init_centroids = init_centroids;
    info.init_labels = init_labels;
    info.total_label_update_time = total_label_update_time;
    info.total_centroid_update_time = total_centroid_update_time;
end


function [cluster_inx, vals, grads, error_flag] = find_closest_centroid(centroids, oracle, X, sparse_ratio, opts)

    error_flag = false;
    
    TF = isnan(centroids);
    if nnz(TF)
        fprintf('# W-kmeans: error: centroids has NAN in find_closest_centroid function.\n');
        error_flag = true;
    end      
        
    k = size(centroids, 1);
    dim = size(centroids, 2);
    n = size(X, 1);

    cluster_inx = zeros(n,1);
    vals = zeros(n,1);
    grads = zeros(n,dim);
    
    if opts.use_parallel
        parfor ii=1:n
            Xii = X(ii,:);

            if opts.sparse_sample
                original_cardinality = nnz(Xii);
                sparse_cardinality = round(original_cardinality * sparse_ratio);
                if strcmp(opts.sparse_alg, 'GSHP')
                    [Xii, ~] = GSHP(Xii, 1, sparse_cardinality);
                elseif strcmp(opts.sparse_alg, 'GSSP') 
                    [Xii, ~] = GSSP(Xii, 1, sparse_cardinality);
                elseif strcmp(opts.sparse_alg, 'Simp') 
                    [Xii, ~] = simple_simplex_sparse(Xii, sparse_cardinality);                
                end
            else
                %Xii = proj(Xii);
            end  

            if isnan(Xii)
                fprintf('# W-kmeans: error: centroids has NAN in find_closest_centroid function.\n');
                error_flag = true;
            else
                dists = zeros(k,1);
                for jj=1:k
                    [D, ~] = oracle(centroids(jj,:), Xii);
                    dists(jj) = D;
                end


                [~, ~] = sort(dists);
                [~, inx] = min(dists);
                cluster_inx(ii) = inx;
                vals(ii) = dists(inx);            
            end
        end  
    else
        for ii=1:n    
            Xii = X(ii,:);

            if opts.sparse_sample
                original_cardinality = nnz(Xii);
                sparse_cardinality = round(original_cardinality * sparse_ratio);
                if strcmp(opts.sparse_alg, 'GSHP')
                    [Xii, ~] = GSHP(Xii, 1, sparse_cardinality);
                elseif strcmp(opts.sparse_alg, 'GSSP') 
                    [Xii, ~] = GSSP(Xii, 1, sparse_cardinality);
                elseif strcmp(opts.sparse_alg, 'Simp') 
                    [Xii, ~] = simple_simplex_sparse(Xii, sparse_cardinality);                
                end
            else
                %Xii = proj(Xii);
            end  

            if isnan(Xii)
                fprintf('# W-kmeans: error: centroids has NAN in find_closest_centroid function.\n');
                error_flag = true;
            else
                dists = zeros(k,1);
                for jj=1:k
                    [D, ~] = oracle(centroids(jj,:), Xii);
                    dists(jj) = D;
                end


                [~, ~] = sort(dists);
                [~, inx] = min(dists);
                cluster_inx(ii) = inx;
                vals(ii) = dists(inx);            
            end
        end
    end
end

