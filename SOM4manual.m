% Sample data (replace with your actual data)
data = load('objectivespace1.csv'); % your data matrix with 3 objective functions

if std(data) > 1
  data = normalize(data);
end

% Define SOM size and create network
som_size = [10 10]; % Adjust based on your data complexity and resolution
som = som_create(som_size, input_dim=size(data,2)); % Use input_dim parameter

% Set learning rate parameters
alpha_0 = 0.5;
decay_rate = 0.01;

% Set training iterations
iterations = 1000;

% SOM training loop
for iter = 1:iterations
  % Decay learning rate
  alpha = alpha_0 * exp(-iter * decay_rate);

  % Train SOM with data samples
  som = som_train(som, data, 'algorithm', 'seq', 'learningrate', alpha);
end

% Map data points onto SOM grid
mapped_data = som_unitdist(som, data);

% Visualization

% Scatter plot (using data objective functions)
figure;
scatter(mapped_data(:,1), mapped_data(:,2), 20, data(:,1), 'filled');
title('Scatter plot on SOM (Objective 1)');
colorbar;
xlabel('SOM X');
ylabel('SOM Y');

% Scatter plot (using other features)
figure;
scatter(mapped_data(:,1), mapped_data(:,2), 20, data(:,2), 'filled');
title('Scatter plot on SOM (Objective 2)');
colorbar;
xlabel('SOM X');
ylabel('SOM Y');

% Scatter plot (using all features)
figure;
scatter(mapped_data(:,1), mapped_data(:,2), 20, data(:,[1 3]), 'filled');
title('Scatter plot on SOM (Objective 1 & 3)');
colorbar;
xlabel('SOM X');
ylabel('SOM Y');

% Parallel coordinates plot
figure;
parallelcoords(data, 'Color', data(:,1)); % Use desired objective function for color
title('Parallel coordinates plot');
xlabel('Objective 1');
ylabel('Objective 2');
zlabel('Objective 3');

% Customize visualizations as needed