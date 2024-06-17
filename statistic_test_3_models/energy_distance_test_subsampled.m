% Energy Distance function with subsampling and manual distance computation
function [stat, p_value] = energy_distance_test_subsampled(X, Y, subsample_size, num_permutations)
    % Subsample the data
    X_sub = subsample_data(X, subsample_size);
    Y_sub = subsample_data(Y, subsample_size);
    
    m = size(X_sub, 1);
    n = size(Y_sub, 1);
    
    % Compute pairwise distances manually
    D_X = pairwise_distances(X_sub, X_sub);
    D_Y = pairwise_distances(Y_sub, Y_sub);
    D_XY = pairwise_distances(X_sub, Y_sub);
    
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
        
        D_perm_X = pairwise_distances(perm_X, perm_X);
        D_perm_Y = pairwise_distances(perm_Y, perm_Y);
        D_perm_XY = pairwise_distances(perm_X, perm_Y);
        
        A_perm = sum(D_perm_X(:)) / (m * m);
        B_perm = sum(D_perm_Y(:)) / (n * n);
        C_perm = sum(D_perm_XY(:)) / (m * n);
        
        perm_stats(i) = 2 * C_perm - A_perm - B_perm;
    end
    
    p_value = mean(perm_stats >= stat);
end