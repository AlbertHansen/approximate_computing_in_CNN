% Read the specific CSV files
acc_epoch = readmatrix("accurate_predictions/1KV8_weights/save_45.csv.csv");
det_epoch = readmatrix("approximate_predictions/1KV8_weights/save_45.csv")';
det_error = det_epoch-acc_epoch;

% Define the folder containing the CSV files
folderPath = './statistical_predictions/1KV8_weights/save_45/';

% Get a list of all CSV files in the folder
fileList = dir(fullfile(folderPath, '*.csv'));

% Initialize an empty cell array to store data from each CSV file
pro_error_cell = cell(1, numel(fileList));

% Loop through each CSV file
for i = 1:numel(fileList)
    % Read the CSV file
    filePath = fullfile(folderPath, fileList(i).name);
    data = readmatrix(filePath); % Assuming the CSV file contains numeric data
    
    % Store the data in the cell array
    pro_error_cell{i} = data - acc_epoch;
end
pro_error = cat(2, pro_error_cell{:});
%%

sorted_pro = sort(pro_error(1,:));
n_pro = numel(sorted_pro);
ecdf_pro = (1:n_pro) / n_pro;



sorted_det = sort(det_error(1,:));
n_det = numel(sorted_det);
ecdf_det = (1:n_det) / n_det;

figure;
stairs(sorted_det, ecdf_det, 'LineWidth', 2);
hold on;
stairs(sorted_pro, ecdf_pro, 'LineWidth', 2);
xlabel('Data Values');
ylabel('EDF');
title('Empirical Distribution Function (EDF)');
grid on;
hold off;
%%

[h,p] = kstest2(det_error(1,1:500),pro_error(1,1:500));
%%

sorted_pro = sort(pro_error(1,:));
n_pro = numel(sorted_pro);
ecdf_pro = (1:n_pro) / n_pro;



sorted_det = sort(det_error(1,:));
n_det = numel(sorted_det);
ecdf_det = (1:n_det) / n_det;

figure;
stairs(sorted_det, ecdf_det, 'LineWidth', 2);
hold on;
stairs(sorted_pro, ecdf_pro, 'LineWidth', 2);
xlabel('Data Values');
ylabel('EDF');
title('Empirical Distribution Function (EDF)');
grid on;
hold off;
%%

sorted_pro = sort(pro_error(1,:));
n_pro = numel(sorted_pro);
ecdf_pro = (1:n_pro) / n_pro;



sorted_det = sort(det_error(1,:));
n_det = numel(sorted_det);
ecdf_det = (1:n_det) / n_det;

figure;
stairs(sorted_det, ecdf_det, 'LineWidth', 2);
hold on;
stairs(sorted_pro, ecdf_pro, 'LineWidth', 2);
xlabel('Data Values');
ylabel('EDF');
title('Empirical Distribution Function (EDF)');
grid on;
hold off;
%%

sorted_pro = sort(pro_error(1,:));
n_pro = numel(sorted_pro);
ecdf_pro = (1:n_pro) / n_pro;



sorted_det = sort(det_error(1,:));
n_det = numel(sorted_det);
ecdf_det = (1:n_det) / n_det;

figure;
stairs(sorted_det, ecdf_det, 'LineWidth', 2);
hold on;
stairs(sorted_pro, ecdf_pro, 'LineWidth', 2);
xlabel('Data Values');
ylabel('EDF');
title('Empirical Distribution Function (EDF)');
grid on;
hold off;

%%
% Concatenate all matrices in the cell array into a single matrix

num_images = numel(pro_error_cell);

% MLE of the observation of probabilistic error
mu_pro = mean(pro_error,2);
covariance_pro = cov(pro_error');
inverse_cov_pro = inv(covariance_pro);

%% 
% Mahalanobis distance of the MLE, to the observed samples
mahalanobis_pro = zeros(1,width(pro_error));

for i = 1:width(pro_error)
    mahalanobis_pro(i) = sqrt((pro_error(:,i)-mu_pro)'*covariance_pro*(pro_error(:,i)-mu_pro));
end

figure;
histogram(mahalanobis_pro,Normalization="pdf");

%% 
sorted_dists = sort(mahalanobis_pro);
n = numel(sorted_dists);
ecdf_pro = (1:n) / n;

figure;
stairs(sorted_dists, ecdf_pro, 'LineWidth', 2);
xlabel('Data Values');
ylabel('EDF');
title('Empirical Distribution Function (EDF)');
grid on;

%% 
% Quantile partitioning for the number of images
% Define the partition points
T = 100; % Number of partitions
partition = linspace(0, 1, T + 1); % Equally spaced partition points

% Calculate the quantiles
quantiles = zeros(1, T);
for j = 1:T
    % Find the smallest Mahalanobis distance such that ECDF >= partition(j)
    idx = find(ecdf_pro >= partition(j), 1, 'first');
    quantiles(j) = sorted_dists(idx);
end

% Plot the ECDF
figure;
plot(sorted_dists, ecdf_pro, 'b-', 'LineWidth', 2);
hold on;

% Plot the quantiles
for j = 1:T
    plot([quantiles(j), quantiles(j)], [0, partition(j)], 'r--', 'LineWidth', 1.5);
end

% Add labels and title
xlabel('Mahalanobis Distance');
ylabel('Cumulative Probability');
title('Empirical Cumulative Distribution Function with Quantiles');

% Add legend
legend('ECDF', 'Quantiles');

% Show grid
grid on;

% Show plot
hold off;

%% 
% Compute Ej = n(pj - pj-1)
expected_counts = zeros(1, T);
for j = 1:T
    expected_counts(j) = num_images * (partition(j+1) - partition(j));
end

%%
% Mahalanobis distance for the deterministic
mahalanobis_det = zeros(1,width(det_error));

for i = 1:width(det_error)
    mahalanobis_det(i) = sqrt((det_error(:,i)-mu_pro)'*covariance_pro*(det_error(:,i)-mu_pro));
end

figure;
histogram(mahalanobis_det,Normalization="pdf");

sorted_det = sort(mahalanobis_det);

appended_quantiles = [quantiles 30];

% Initialize a vector to store the counts
observed_counts = zeros(1, T);

% Count the number of indices in each interval between quantiles
for j = 1:T
    % Logical indexing to find indices between quantiles
    indices_between_quantiles = sorted_det >= appended_quantiles(j) & sorted_det < appended_quantiles(j+1);
    % Count the number of indices
    observed_counts(j) = sum(indices_between_quantiles);
end

%%
% Compute A_T statistic
A_t = 0;
for i = 1:T
    A_t = A_t + (abs(expected_counts(i)-observed_counts(i))/expected_counts(i));
end
%A_t = A_t/T;

%%
% Number of samples 
% Empirical p-value through simulation using the Y system's empirical distribution
B = 100;
A_T_simulated = zeros(1, B);
for b = 1:B
    % Number of samples you want to generate
    num_samples = 1000;
    sim_sample = pro_error(:,randi(size(pro_error, 2), num_samples, 1));

    sim_mean = mean(sim_sample');
    sim_cov = cov(sim_sample');
    sim_inv_cov = inv(sim_cov);
    
    mahalanobis_sim = zeros(1,width(sim_sample));

    for i = 1:width(sim_sample)
        mahalanobis_sim(i) = sqrt((sim_sample(:,i)-mu_pro)'*covariance_pro*(sim_sample(:,i)-mu_pro));
    end

    sorted_sim = sort(mahalanobis_sim);

    % Initialize a vector to store the counts
    sim_observed_counts = zeros(1, T);

    % Count the number of indices in each interval between quantiles
    for j = 1:T
        % Logical indexing to find indices between quantiles
        indices_between_quantiles = sorted_sim >= appended_quantiles(j) & sorted_sim < appended_quantiles(j+1);
        % Count the number of indices
        sim_observed_counts(j) = sum(indices_between_quantiles);
    end

    A_t_sim = 0;
    for i = 1:T
        A_t_sim = A_t_sim + (abs(expected_counts(i)-sim_observed_counts(i))/expected_counts(i));
    end
    A_T_simulated(b) = A_t_sim/T;
end

%%

% Compute the indicator function for each simulation
indicator_values = A_t_sim >= A_t;

% Calculate p_e as the mean of the indicator values
p_e = mean(indicator_values);