data = load('objectivespace1.csv');

% Preprocess data (assuming numerical features)
data = mapminmax(data);

% Create and train SOM
som_size = [10 10];  % Adjust the grid size as needed
net = selforgmap(som_size);
net.trainParam.epochs = 500;
[net, tr] = train(net, data);

% Visualize (e.g., component planes)
figure;
plotsomplanes(net);

% Get cluster assignments
neuron_mapping = vec2ind(net(data))';

% Analyze specific cluster (e.g., cluster 1)
cluster1_indices = find(neuron_mapping == 1);
cluster1_data = data(cluster1_indices, :);

% ... explore features, perform further analysis

% Example of saving the trained SOM
save('my_som_model.mat', 'net');