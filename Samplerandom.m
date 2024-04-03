% Define your data set (replace with your actual data)
data = [randn(100,2); randn(100,2)+2];  % Sample data with two features

% Define the SOM grid size
gridSize = 10;

% Create a self-organizing map object
net = selforgmap([gridSize gridSize]);

% Train the SOM network with the data
[net, tr] = train(net, data);

% Get the winning neuron indices for each data point
winning_neurons = reshape(net(data), [], size(net.layers{1}.dimensions,1));

% Define the number of components to visualize (same as data features)
componentsToPlot = size(data,2);  

% Create a new figure for visualization
figure;

% Loop through each data point and its winning neuron
% Loop through each data point and its winning neuron (handle zeros)
for i = 1:size(data,1)
  winning_neuron = winning_neurons(i,:);
  if any(winning_neuron == 0)  % Check for zeros
    component_values = zeros(size(net.IW{1},1),1);  % Use zeros vector if all zeros
  else
    component_values = net.IW{1}(:,winning_neuron);
  end
  % ... (rest of your code using component_values)
end

% ... (rest of your code)

% Customize the plot further using markers, colors, and legends
xlabel('Component 1');
ylabel('Component 2');  % Adjust labels for higher dimensions
title('SOM Component Planes for Data Points');

hold off;