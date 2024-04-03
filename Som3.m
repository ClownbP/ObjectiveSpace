data = load('objectivespace1.csv');

% Preprocess data (assuming numerical features)
data = mapminmax(data);


% Learning rate and neighborhood size functions (decreasing)
learning_rate = @(t) 0.5 * (1 - t);
neighborhood_size = @(t) max(2, sum(map_size) - 2 * t);
map_size = [10 10];
epochs=100;

% Initialize weights randomly
weights = rand(map_size(1), map_size(2), size(data, 2));

% Train the SOM using a loop
for epoch = 1:epochs
  for i = 1:size(data, 1)
    datapoint = data(i, :);

    % Find the Best Matching Unit (BMU)
    [~, winner] = min(sum(bsxfun(@minus, datapoint, weights).^2, 2));

    % Update weights using Gaussian neighborhood function
    for i_map = 1:map_size(1)
      for j_map = 1:map_size(2)
        distance = sqrt(sum((winner - [i_map j_map]).^2));
        if distance <= neighborhood_size(epoch)
          influence = exp(-distance^2 / (2 * neighborhood_size(epoch)^2));
          weights(i_map, j_map, :) = weights(i_map, j_map, :) + ...
                                     learning_rate(epoch) * influence * (datapoint - weights(i_map, j_map, :));
        end
      end
    end
  end
end

% Calculate the U-matrix
u_matrix = zeros(map_size);
for i = 1:map_size(1)
  for j = 1:map_size(2)
    if i < map_size(1)
      u_matrix(i, j) = u_matrix(i, j) + sqrt(sum((weights(i, j, :) - weights(i + 1, j, :)).^2));
    end
    if j < map_size(2)
      u_matrix(i, j) = u_matrix(i, j) + sqrt(sum((weights(i, j, :) - weights(i, j + 1, :)).^2));
    end
  end
end

% Plot the U-matrix
imagesc(u_matrix);
colorbar;
title('U-Matrix for SOM with 3-Variable Data');
xlabel('X-axis');
ylabel('Y-axis');