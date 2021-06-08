clear;
clc;
close all;


%% set params
dataset_name = 'COIL20';
%dataset_name = 'COIL20';
%dataset_name = 'COIL100';
%dataset_name = 'CIFAR100';
%dataset_name = 'MNIST';
%dataset_name = 'USPS';
%dataset_name = 'ALOI';
hist_dim = 1;
%hist_dim = 2;
max_iter = 10;

eval_non_sparse_flag = false;


%% load dataset
if strcmp(dataset_name, 'COIL20')
    input = importdata('dataset/COIL20_probability_new.mat');
    if hist_dim == 1
        X = input.AllSetIntensityProbabilityWithZero;
    else
        X = input.AllSet2d16x16IntensityProbability;
    end    
    gnd = input.AllSet.y;
elseif strcmp(dataset_name, 'COIL100')
    input = importdata('dataset/COIL100_probability_new.mat');
    if hist_dim == 1
        X = input.AllSetIntensityProbabilityWithZero;
    else
        X = input.AllSet2d16x16IntensityProbability;
    end        
    gnd = input.AllSet.y;
elseif strcmp(dataset_name, 'CIFAR100')
    input = importdata('dataset/CIFAR100_probability.mat');
    X = input.AllSetProbability;    
    gnd = input.AllSet.y;  
elseif strcmp(dataset_name, 'MNIST')    
    input = importdata('dataset/MNIST_probability_new.mat');    
    if hist_dim == 1
        %input = importdata('dataset/MNIST_probability.mat');
        X = input.AllSetIntensityProbabilityWithZero;
    else
        X = input.AllSet2d14x14IntensityProbability;
    end
    gnd = input.AllSet.y+1;
elseif strcmp(dataset_name, 'USPS')    
    input = importdata('dataset/USPS_probability.mat');    
    if hist_dim == 1
        X = input.AllSetIntensityProbabilityWithZero;
    else
        X = input.AllSet2d16x16IntensityProbability;
    end
    gnd = input.AllSet.y;    
elseif strcmp(dataset_name, 'ALOI')    
    input = importdata('dataset/ALOI/ALOI_probability.mat');
    %X = input.AllSetProbabilityWithZero_27;
    %X = input.AllSetProbabilityWithZero_64;
    %X = input.AllSetProbabilityWithZero_125;
    X = input.AllSetProbabilityWithZero_216;

    if 0
        dim = size(X,1);
        X = X(2:dim,:);
        for ii=1:size(X,2)  
            Xii = X(:,ii);

            sparse_alg = 'GSSP';
            sparse_ratio = 1.0;

            original_cardinality = nnz(Xii);
            sparse_cardinality = round(original_cardinality * sparse_ratio);
            if strcmp(sparse_alg, 'GSHP')
                [Xii, ~] = GSHP(Xii, 1, sparse_cardinality);
            elseif strcmp(sparse_alg, 'GSSP') 
                [Xii, ~] = GSSP(Xii, 1, sparse_cardinality);
            elseif strcmp(sparse_alg, 'Simp') 
                [Xii, ~] = simple_simplex_sparse(Xii, sparse_cardinality);                
            end

            X(:,ii) = Xii;
        end
    end
    gnd = input.gnd';    
end

classnum = input.class_num;
clear input

%% manipulate dataset
if 1
    max_train_samples = 3;
    classnum = 10;
    %max_train_samples = 2;
    %classnum = 2;    
    X_red = [];
    gnd_red = [];

    for i=1:classnum
        class_index = find(gnd == i);
        if length(class_index) > max_train_samples
            %class_index = class_index(1: max_samples);
            class_index_rnd = randperm(length(class_index));
            class_index_tmp = class_index(class_index_rnd);
            class_index = class_index_tmp(1: max_train_samples);
        end
        X_red = [X_red X(:, class_index)];
        gnd_red = [gnd_red gnd(1, class_index)];

    end

    X = X_red;
    gnd = gnd_red';

elseif 0
    max_train_samples = 50;
    X_red = [];
    gnd_red = [];


    class_idx = [16, 20];
    %class_idx = [16, 18, 20];
    %class_idx = [1, 3, 5, 8, 10, 11, 12, 15, 16, 18, 20];
    classnum = length(class_idx);
    for i=1:classnum
        class_index = find(gnd == class_idx(i));
        if length(class_index) > max_train_samples
            %class_index = class_index(1: max_samples);
            class_index_rnd = randperm(length(class_index));
            class_index_tmp = class_index(class_index_rnd);
            class_index = class_index_tmp(1: max_train_samples);
        end
        X_red = [X_red X(:, class_index)];
        gnd_red = [gnd_red ones(1,length(class_index))*i];

    end

    X = X_red;
    gnd = gnd_red';

else
    gnd = gnd';
end

X = double(X); 
%X = X / 255;

sample_num = size(X,2);
dim_num = size(X,1);

%% purmutate
inx = randperm(sample_num);
X = X(:, inx);
gnd = gnd(inx);
X = X';

fprintf('### %s: class=%d, sample=%d, dim=%d, max_iter=%d, hist_dim=%d\n', dataset_name, classnum, sample_num, dim_num, max_iter, hist_dim);



%% Calculate cost matrix C
%C = build_1d_hist_pairwise_distance_matrix(dim_num);
C = build_hist_pairwise_distance_matrix(hist_dim, dim_num);
C = C / max(max(C));


%% Euclidean kmeans (litekmeans)
start_time = tic();
[init_labels, init_centroids] = litekmeans(X, classnum, 'Replicates', 20);
elapsed_time = toc(start_time);
[accuracy] = eval_clustering_accuracy(gnd, init_labels);
fprintf('====== Euclidean kmeans\n');
fprintf('# Euclidean kmeans (litekmeans): purity=%.4f, nmi=%.4f, acc=%.4f\n\n', ...
        accuracy.purity, accuracy.nmi, accuracy.acc);  
     

%% Wasserstein k-means
% set options
opts = [];
opts.verbose = 3;
opts.lambda = 0.1;
opts.k = classnum;
opts.init_alg = 'kmeans';
opts.p = 1; % p-Wasserstein
opts.sinkhorn_lambda = 1e-2;
opts.gnd = gnd;
opts.max_iter = max_iter;
opts.use_parallel = false;

%opts.OTSolver = OTSolver.Gurobi;
opts.OTSolver = OTSolver.Linprog;
%opts.OTSolver = OTSolver.FastEMD; 
%opts.OTSolver = OTSolver.Sinkhorn;
%opts.OTSolver = OTSolver.Mosek;


% evaluate non-sparse setting
if eval_non_sparse_flag

    opts.barycenter_alg = 'sinkhorn';
    opts.sparse_centroid = false;
    opts.sparse_sample = false;
    opts.sparse_ratio = 0;
    opts.sparse_ratio_alg = 'nan'; % non-available
    opts.sparse_alg = 'nona';
    fprintf('====== Wasserstein kmeans [%d/%d-%d] (com:%d, cent:%d, samp:%d, alg:%s, ratio_alg:%s, ratio:%.2f, bary:%s)\n', ...
        eval_iter, evaluations, config_index, opts.compress, opts.sparse_centroid, opts.sparse_sample, opts.sparse_alg, opts.sparse_ratio_alg, opts.sparse_ratio, opts.barycenter_alg); 

    [centroids, infos] = sspw_kmeans(C, X, init_centroids, init_labels, opts);
    init_labels = infos.init_labels;
    init_centroids = infos.init_centroids;

    [accuracy] = eval_clustering_accuracy(gnd, infos.labels);
    fprintf('# Wasserstein kmeans: purity=%.4f, nmi=%.4f, acc=%.4f, time=%.1f[sec]\n\n', ...
        accuracy.purity, accuracy.nmi, accuracy.acc, infos.total_label_update_time + infos.total_centroid_update_time); 
end



opts.compress = true;
opts.sparse_ratio = 0.6;
opts.sparse_ratio_alg = 'fix'; % {'fix','dec','inc'};
opts.sparse_alg = 'GSSP'; % {'Simp', 'GSHP', 'GSSP'};
opts.sparse_centroid = 1;
opts.sparse_sample = 1; 
opts.barycenter_alg = 'sinkhorn'; % {'sinkhorn', 'stabilized'};               

if ~opts.sparse_centroid && ~opts.sparse_sample
    return;
end


fprintf('====== Wasserstein kmeans (com:%d, cent:%d, samp:%d, alg:%s, ratio_alg:%s, ratio:%.2f, bary:%s)\n', ...
    opts.compress, opts.sparse_centroid, opts.sparse_sample, opts.sparse_alg, opts.sparse_ratio_alg, opts.sparse_ratio, opts.barycenter_alg); 

%rng('default')
[centroids, infos] = sspw_kmeans(C, X, init_centroids, init_labels ,opts);

[accuracy] = eval_clustering_accuracy(gnd, infos.labels);
fprintf('# Wasserstein kmeans: purity=%.4f, nmi=%.4f, acc=%.4f, time=%.1f[sec]\n\n', ...
    accuracy.purity, accuracy.nmi, accuracy.acc, infos.total_label_update_time + infos.total_centroid_update_time);           

