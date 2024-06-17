% Energy Distance function with subsampling
function [stat, p_value] = energy_distance_test_subsampled(X, Y, subsample_size, num_permutations)
    % Subsample the data
    X_sub = subsample_data(X, subsample_size);
    Y_sub = subsample_data(Y, subsample_size);
    
    m = size(X_sub, 1);
    n = size(Y_sub, 1);
    
    % Compute pairwise distances
    D_X = pdist2(X_sub, X_sub, 'euclidean');
    D_Y = pdist2(Y_sub, Y_sub, 'euclidean');
    D_XY = pdist2(X_sub, Y_sub, 'euclidean');
    
    % Compute the Energy Distance statistic
    A = sum(D_X(:)) / (m * m);
    B = sum(D_Y(:)) / (n * n);
    C = sum(D_XY(:)) / (m * n);
    
    stat = 2 * C - A - B;
    
    % Permutation test for p-value
    combined = [X_sub; Y_sub];
    perm_stats = zeros(num_permutations, 1);
    
    for i = 1:num_permutations
        perm_indices = randperm(m + n);
        perm_X = combined(perm_indices(1:m), :);
        perm_Y = combined(perm_indices(m+1:end), :);
        
        D_perm_X = pdist2(perm_X, perm_X, 'euclidean');
        D_perm_Y = pdist2(perm_Y, perm_Y, 'euclidean');
        D_perm_XY = pdist2(perm_X, perm_Y, 'euclidean');
        
        A_perm = sum(D_perm_X(:)) / (m * m);
        B_perm = sum(D_perm_Y(:)) / (n * n);
        C_perm = sum(D_perm_XY(:)) / (m * n);
        
        perm_stats(i) = 2 * C_perm - A_perm - B_perm;
    end
    
    p_value = mean(perm_stats >= stat);
end