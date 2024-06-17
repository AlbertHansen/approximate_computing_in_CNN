% Subsampling function
function subsample = subsample_data(data, subsample_size)
    indices = randperm(size(data, 1), subsample_size);
    subsample = data(indices, :);
end