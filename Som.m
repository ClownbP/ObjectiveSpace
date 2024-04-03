%%----------------SOM Clustering --------------------------
% -------------------- Code -------------------------------

clear;

rng('default');

% Define predictors
predictors = [1210849.418	541215.041	37.64269983; 1338266.311	430301.846	40.91041687; 2178772.558	455430.2394	42.43608621; 1192456.635	605808.241	38.14696364; 1514452.357	396947.9041	38.63927889; 1186131.178	902022.244	41.29770366];
x = predictors;


% Create a Self-Organizing Map
dimension1 = 7;
dimension2 = 7;
net = selforgmap([dimension1 dimension2]);
net.trainParam.epochs = 500;


% Train the Network
[net,tr] = train(net,x);

% Test the Network
y = net(x);


% View the Network
view(net)

% Plots
% Uncomment these lines to enable various plots.
%figure, plotsomtop(net)
%figure, plotsomnc(net)
%figure, plotsomnd(net)
%figure, plotsomplanes(net)
%figure, plotsomhits(net,x)
%figure, plotsompos(net,x)


% Fetch data points
input_neuron_mapping = vec2ind(net(x))';
neuron_1_input_indices = find(input_neuron_mapping == 1)
neuron_1_input_values = x(neuron_1_input_indices)