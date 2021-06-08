function [C] = build_hist_pairwise_distance_matrix(dim, hist_size)


    %% build the pairwise distance matrix
    
    if dim == 1
        x = [1:hist_size] - 1;
        x = x';

        X = x.*x;
        Y = X';
        C = -2 * (x * x') + repmat(X,[1,hist_size]) + repmat(Y,[hist_size,1]);
        C = sqrt(C);
    
    elseif dim == 2
        
        width = sqrt(hist_size);
        height = sqrt(hist_size);        

        x = 1:width;
        y = 1:height;

        ordered_pairs = zeros(width*height, 2);
        ordered_pairs(:,2) = repmat(y, 1, height);
        for ii=1:height
            ordered_pairs(width*(ii-1) + (1:width),1) = x(ii);
        end

        d2 = @(a,b) sqrt((a(1) - b(1))^2 + (a(2) - b(2))^2);

        C = zeros(width*height, width*width);
        for ii=1:width*height
            for jj=1:width*height
                C(ii,jj) = d2(ordered_pairs(ii,:), ordered_pairs(jj,:));
            end
        end
        
    else
        
    end

end
