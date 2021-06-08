function [new, keep_nnz_index] = simple_simplex_sparse(original, cardinality)
    %nnz_index = find(original>0);
    %nnz_w = original(nnz_index);
    [~, descend_index] = sort(original, 'descend');
    keep_nnz_index = descend_index(1:cardinality);
    new = original;
    index = false(length(original),1);
    index(keep_nnz_index) = 1;
    new(~index) = 0;
    
    new = new * 1/sum(new);
end