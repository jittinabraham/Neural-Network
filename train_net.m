function [network2, error] = train_net(network, X, R, max_epochs, alpha)
    [~, length] = size(X);
    output = zeros(size(R));
    threshold = 9.87e-8;
    
    for epoch = 1:max_epochs
        epoch
        for i = 1:length
            [output(:, i),A] = forward_pass(X(:, i), network);
            
            network = back_pass(network, output(:, i), alpha, R(:, i), A);
        end
        
        % Calculate error (mean squared error)
        error(epoch) = mean(mean((output - R).^2));
        
         %Check stopping criterion (e.g., based on error)
       if error(epoch) < threshold
            break;
        end
    end
    
    network2 = network;
end

function [output,A] = forward_pass(x, network, A)
    numLayers = numel(network);
    A = struct();
    A(1).value=x;
    for i = 2:numLayers
        weights = network(i).W;
        bias = network(i).b;
        z = weights * x + bias;
        x = sigmoid(z);
        A(i).value = x;
        
    end
    A(end).value;
    
    output = x;
end

function network = back_pass(network, output, alpha, target, A)
    numLayers = numel(network);
    A(end).value;
    output;
    delta = (output - target) .* sigmoid_derivative(A(end).value);
    
    for i = numLayers:-1:2
        
        weights = network(i).W;
        
        activations_prev = sigmoid_derivative(A(i-1).value);
       
        delta_prev = delta;%%%10 by 1
        
       
        delta = (weights' * delta_prev) .* activations_prev;%%%50 byy 1
   
       
        % Weight update
        network(i).W = network(i).W - (alpha * delta_prev * activations_prev');
        network(i).b = network(i).b - (alpha * delta_prev);
    end
end

function output = sigmoid(x)
    output = 1 ./ (1 + exp(-x));
end

function output = sigmoid_derivative(x)
    output = sigmoid(x) .* (1 - sigmoid(x));
end
