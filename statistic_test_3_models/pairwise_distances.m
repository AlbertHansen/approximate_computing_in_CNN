% Function to calculate pairwise Euclidean distances manually
function D = pairwise_distances(X, Y)
    m = size(X, 1);
    n = size(Y, 1);
    D = zeros(m, n);
    for i = 1:m
        for j = 1:n
            D(i, j) = sqrt(sum((X(i, :) - Y(j, :)).^2));
        end
    end
end