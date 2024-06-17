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
    pro_error_cell{i} = mean((data - acc_epoch),2);
end

pro_error = cat(2, pro_error_cell{1:500});

%%
det_error = det_error(:,1:500);
num_images = numel(pro_error_cell);

% MLE of the observation of probabilistic error
mu_pro = mean(pro_error,2);
covariance_pro = cov(pro_error');
inverse_cov_pro = inv(covariance_pro);

%%


%%
figure;
histogram(det_error(1,:), Normalization="probability");
hold on;
histogram(pro_error(1,:), Normalization="probability");
hold off;
figure;
histogram(det_error(2,:), Normalization="probability");
hold on;
histogram(pro_error(2,:), Normalization="probability");
hold off;
figure;
histogram(det_error(3,:), Normalization="probability");
hold on;
histogram(pro_error(3,:), Normalization="probability");
hold off;
figure;
histogram(det_error(4,:), Normalization="probability");
hold on;
histogram(pro_error(4,:), Normalization="probability");
hold off;
figure;
histogram(det_error(5,:), Normalization="probability");
hold on;
histogram(pro_error(5,:), Normalization="probability");
hold off;
%%
figure;
histogram(det_error(6,:), Normalization="probability");
hold on;
histogram(pro_error(6,:), Normalization="probability");
hold off;
figure;
histogram(det_error(7,:), Normalization="probability");
hold on;
histogram(pro_error(7,:), Normalization="probability");
hold off;
figure;
histogram(det_error(8,:), Normalization="probability");
hold on;
histogram(pro_error(8,:), Normalization="probability");
hold off;
figure;
histogram(det_error(9,:), Normalization="probability");
hold on;
histogram(pro_error(9,:), Normalization="probability");
hold off;
figure;
histogram(det_error(10,:), Normalization="probability");
hold on;
histogram(pro_error(10,:), Normalization="probability");
hold off;
%%
% Example usage with debug prints

% Call the test function with debugging
%[p, e_n, e_n_boot] = minentest(det_error(1:100, :), pro_error(1:100, :));
%%

[h,p] = ttest2(det_error(2,:)', pro_error(2,:)',"Vartype","unequal");
%%

