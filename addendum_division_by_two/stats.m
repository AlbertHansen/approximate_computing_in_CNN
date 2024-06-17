epochs = [45 50 55 60 65 70 75 80 85 ];
KL_divergences = zeros(1, numel(epochs));
for epoch = 1:numel(epochs)
    epoch_value = epochs(epoch);
    % Read the specific CSV files

    filename_acc = sprintf("accurate_predictions/1KV8_weights/save_%d.csv.csv", epoch_value);
    filename_det = sprintf("approximate_predictions/1KV8_weights/save_%d.csv", epoch_value);
    acc_epoch = readmatrix(filename_acc);
    det_epoch = readmatrix(filename_det)';
    det_error = det_epoch-acc_epoch;

    % Define the folder containing the CSV files
    folderPath = sprintf('./statistical_predictions/1KV8_weights/save_%d/',epoch_value);

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
        %pro_error_cell{i} = data - acc_epoch;
        pro_error_cell{i} = data - acc_epoch;
    end
    pro_error = cat(2, pro_error_cell{:});

   
    outputFolder = sprintf('KV8_%d',epoch_value);
    if ~exist(outputFolder, 'dir')
        mkdir(outputFolder);
    end
    for i = 1:10
        f = figure(Position=[10 10 300 250]); % Create a new figure
        histogram(det_error(i, :), 20, 'Normalization', 'probability');
        hold on;
        histogram(pro_error(i, :), 50, 'Normalization', 'probability');
        title(['Error Variable ' num2str(i)], 'Interpreter', 'latex');
        xlabel('Error', 'Interpreter', 'latex');
        xlim([-2.6 .2]);
        ylim([0 .2]);
        ylabel('Probability', 'Interpreter', 'latex');
        legend({'Approximate Model', 'Probabilistic Model'}, 'Interpreter', 'latex');
        grid on;
        hold off;

        % Construct the file name
        fileName = fullfile(outputFolder, [sprintf('KV8_%d_',epoch_value) num2str(i) '.pdf']);

        % Export the figure as a PDF using exportgraphics
        exportgraphics(f, fileName, 'ContentType', 'vector'); % Export to PDF with vector graphics

        % Close the figure to free up memory
        close(f);
    end
    % Parameters
    lambda = 0.00001; % Regularization parameter, adjust as needed

    % Compute the means and covariance matrices
    mu = mean(pro_error, 2);
    Sigma = cov(pro_error');

    sample_mean = mean(det_error, 2);
    sample_covariance = cov(det_error');

    % Regularize the covariance matrices
    Sigma_regularized = Sigma + lambda * eye(size(Sigma));
    sample_covariance_regularized = sample_covariance + lambda * eye(size(sample_covariance));

    % Ensure Sigma_regularized is invertible
    Sigma_inv = inv(Sigma_regularized);
    k = length(mu);
    % Compute the terms of the KL divergence formula with regularized covariance matrices
    trace_term = trace(Sigma_inv * sample_covariance_regularized)-k;
    mean_diff = sample_mean - mu;
    mean_diff_term = mean_diff' * Sigma_inv * mean_diff;
    log_det_term = log(det(Sigma_regularized) / det(sample_covariance_regularized));

    % Compute the KL divergence
    KL_divergence = 0.5 * (trace_term + mean_diff_term + log_det_term);

    KL_divergences(epoch) = KL_divergence;


end
% Specify the filename for the CSV file
filename = 'KL_divergences_KV8_3_model_stat.csv';

% Write the vector to the CSV file
writematrix(KL_divergences, filename);

%%
